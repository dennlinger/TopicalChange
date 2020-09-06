"""
Extract hierarchical section information from HTML content and store them as
*.json files.
"""

from bs4 import BeautifulSoup, NavigableString, Comment
from langid.langid import LanguageIdentifier, model
from multiprocessing import Pool
import collections
import copy
import sys
import re

re_header = re.compile(
    """
^(?:[(]?
    (?:[0-9]+|
       [a-zA-Z]|
       [ivx]+|
       [IVX]+)
    [.):]|

    (?:PART|part|Part|SECTION|section|Section|ARTICLE|article|Article)
    \\s+
    (?:[0-9]+|
       [a-zA-Z]|
       [ivx]+|
       [IVX]+)
    [.):]|

    [0-9]+[ ])
(.*)$""",
    re.VERBOSE,
)


class WebsiteNotStructured(BaseException):
    pass


def parent_tags(tag):
    """ Return a list of all parent tags of a given node. """
    tags = []
    while tag:
        tags.append(tag.name)
        tag = tag.parent
    return tags


def convert_pres(soup):
    """ Convert <pre>...</pre> to several <p> tags (one for each line). """
    while True:
        block = soup.find("pre")
        if block is None:
            break

        content = "".join([str(c) for c in block.contents])

        for line in reversed(content.split("\n")):
            new_tag = soup.new_tag("p")
            new_tag.append(NavigableString(line))
            block.insert_after(new_tag)

        block.extract()


def convert_spans(soup):
    """ Convert <span>...</span> tags to <b>...</b> or <u>...</u> tags. """
    while True:
        block = soup.find("span")
        if block is None:
            break

        if re.match(
            "^(.*;\\s*)?font-weight\\s*:\\s*bold(;.*)?$", block.get("style", "")
        ):
            if "b" not in parent_tags(block):
                block.name = "b"
                continue
            # Otherwise delete it without replacement.

        if re.match(
            "^(.*;\\s*)?text-decoration\\s*:\\s*underline(;.*)?$",
            block.get("style", ""),
        ):
            if "u" not in parent_tags(block):
                block.name = "u"
                continue
            # Otherwise delete it without replacement.

        if "underline" in block.get("class", []):
            if "u" not in parent_tags(block):
                block.name = "u"
                continue
            # Otherwise delete it without replacement.

        for child in reversed(block.contents):
            child = child.extract()
            block.insert_after(child)

        block.extract()


def delete_divs(soup):
    """ Delete all <div>...</div> tags. """
    while True:
        block = soup.find("div")
        if block is None:
            break

        for child in reversed(block.contents):
            child = child.extract()
            block.insert_after(child)

        block.extract()


def is_string(tag):
    """ Checks if a node is a string without proper tag. """
    for child in tag.contents:
        if isinstance(child, NavigableString):
            if len(str(child).strip()) > 0:
                break
        if child.name in ["strong", "b", "u"]:
            break
    else:
        return False

    while True:
        if tag.name in ["style", "script", "title"]:
            return False
        if tag.name in ["p", "h1", "h2", "h3", "h4", "h5"]:
            return False
        tag = tag.parent
        if tag is None:
            break

    return True


def convert_strings(soup):
    """ Put <p>...</p> tags around all strings without proper tags. """
    while True:
        block = soup.find(is_string)
        if block is None:
            break

        new_tag = soup.new_tag("p")

        for child in list(block.contents):
            child = child.extract()
            new_tag.append(child)

        block.append(new_tag)


def is_consecutive(tag):
    """ Check if several nodes of same type follow each other. """
    if tag.name not in ["b", "strong", "u"]:
        return False

    next_tag = tag.next_sibling
    while True:
        if not next_tag:
            return False
        if not isinstance(next_tag, NavigableString):
            break
        if len(str(next_tag).strip()) > 0:
            return False
        next_tag = next_tag.next_sibling

    return next_tag.name == tag.name


def merge_consecutive(soup):
    """ Merge consecutive nodes if they have the same type. """
    while True:
        block = soup.find(is_consecutive)
        if block is None:
            break

        while True:
            next_block = block.next_sibling
            assert next_block is not None
            if not isinstance(next_block, NavigableString):
                break
            child = next_block.extract()
            block.append(child)

        assert next_block.name == block.name
        for child in list(next_block.contents):
            child = child.extract()
            block.append(child)

        next_block.extract()


def is_nested(tag, tag_list):
    if tag.name in tag_list:
        return tag.find(tag_list)
    return False


def split_on_nested(soup, tag_list):
    """ Split nested tags (e.g., nested <p> tags) into separate ones. """
    while True:
        block = soup.find(lambda tag: is_nested(tag, tag_list))
        if block is None:
            break

        split = block.find(tag_list)
        assert split is not None

        while True:
            split_parent = split.parent
            new_tag = copy.copy(split_parent)
            new_tag.clear()

            index = split_parent.index(split)
            for element in split_parent.contents[index + 1 :]:
                element = element.extract()
                new_tag.append(element)

            split = split.extract()
            if len(new_tag.contents) > 0:
                split_parent.insert_after(new_tag)
            if split_parent == block:
                split_parent.insert_after(split)
                if len(split_parent.contents) == 0:
                    split_parent.extract()
                break

            new_tag = copy.copy(split_parent)
            new_tag.clear()

            for element in list(split.contents):
                element = element.extract()
                new_tag.append(element)

            split.append(new_tag)
            split_parent.insert_after(split)
            if len(split_parent.contents) == 0:
                split_parent.extract()


def is_splittable_block(tag, block_tag, split_tag):
    if tag.name == block_tag:
        return tag.find(split_tag)
    return False


def split_on_tag(soup, block_tag, split_tag, attr=None):
    """ Split HTML tree blocks based on split tag. """
    while True:
        block = soup.find(lambda tag: is_splittable_block(tag, block_tag, split_tag))
        if block is None:
            break

        split = block.find(split_tag)
        assert split is not None

        while True:
            split_parent = split.parent
            new_tag = copy.copy(split_parent)
            new_tag.clear()

            index = split_parent.index(split)
            for element in split_parent.contents[index + 1 :]:
                element = element.extract()
                new_tag.append(element)

            split = split.extract()
            split_parent.insert_after(new_tag)
            if split_parent == block:
                if attr is not None:
                    new_tag[attr] = re.sub("\\s+", " ", split.get_text()).strip()
                break

            split_parent.insert_after(split)


def extract_index(text):
    """ Clean up text and extract nesting level. """
    count = 0
    text = text.replace("\u200b", "")
    text = text.replace("\u00ad", "-")
    text = text.strip()
    while True:
        if len(text) > 0 and text[0] in ["•", "-", "–"]:
            text = text[1:].strip()
            continue

        m = re_header.match(text)
        if m is None:
            break
        text = m.group(1).strip()
        count += 1

    if len(text) > 0 and text[0] in [".", ","]:
        text = text[1:].strip()
    return text, count


def is_bold(tag):
    return tag.name in ["strong", "b"]


def is_underline(tag):
    return tag.name in ["u"]


def split_tag_enum(tag, func, count):
    """
    Helper function for split_on_tag to split based on bold / unterline tags
    containing an enumeration.
    """

    if not func(tag):
        return False
    if tag.parent.name != "p":
        return False
    if tag.previous_sibling:
        return False

    text = re.sub("\\s+", " ", tag.get_text())
    _, index = extract_index(text)

    if count != index:
        return False
    if len(text) > 200:
        return False

    return True


def split_text_enum(tag, count):
    """
    Helper function for split_on_tag to split based on text containing an
    enumeration.
    """

    if tag.name != "p":
        return False
    if not tag.next_sibling:
        return False
    if tag.next_sibling.name != "p":
        return False

    text = re.sub("\\s+", " ", tag.get_text())
    text, index = extract_index(text)

    if count != index:
        return False
    if len(text) > 100:
        return False

    text = re.sub("\\s+", " ", tag.next_sibling.get_text())
    _, index = extract_index(text)

    if index != 0:
        return False

    return True


def split_tag_list(tag, func, count):
    """
    Helper function for split_on_tag to split based on an enumeration with
    <li> tags.
    """

    if not func(tag):
        return False
    if tag.parent.name != "p":
        return False
    if tag.previous_sibling:
        prev = tag.previous_sibling
        if not isinstance(prev, NavigableString):
            return False
        text, _ = extract_index(str(prev))
        if len(text) > 0:
            return False

    index = 0
    current = tag.parent
    while current:
        if current.name == "li":
            index += 1
        current = current.parent

    if count != index:
        return False

    return True


def extract_content(html, headless=True, language="en"):
    import boilerpipe

    # Workaround for BeautifulSoup causing a RecursionError.
    sys.setrecursionlimit(10000)

    try:
        soup = BeautifulSoup(html, "html.parser")
    except NotImplementedError:
        return []  # Bug in BS4?

    convert_pres(soup)
    convert_spans(soup)
    merge_consecutive(soup)
    split_on_tag(soup, "p", "br")
    html = str(soup)

    html = boilerpipe.extract(html)
    soup = BeautifulSoup(html, "html.parser")
    assert len(soup.contents) == 1
    assert soup.contents[0].name == "html"
    convert_strings(soup)
    split_on_nested(soup, ["p", "h1", "h2", "h3", "h4", "h5"])
    delete_divs(soup)

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    scores = collections.defaultdict(float)
    for paragraph in soup.find_all("p"):
        text = re.sub("\\s+", " ", paragraph.get_text()).strip()
        try:
            lang, conf = identifier.classify(text)
        except UnicodeEncodeError:
            continue
        scores[lang] += len(text) * conf

    if len(scores) == 0:
        return []
    lang, _ = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0]
    if lang != language:
        return []

    for tag in ["h1", "h2", "h3", "h4", "h5"]:
        attr = "section-%s" % (tag,)
        temp = copy.copy(soup)
        split_on_tag(temp, "html", tag, attr=attr)
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    for count in range(1, 6):
        attr = "section-b%d" % (count,)
        temp = copy.copy(soup)
        split_on_tag(
            temp, "html", lambda tag: split_tag_enum(tag, is_bold, count), attr=attr
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    for count in range(1, 6):
        attr = "section-li-b%d" % (count,)
        temp = copy.copy(soup)
        split_on_tag(
            temp, "html", lambda tag: split_tag_list(tag, is_bold, count), attr=attr
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    if len(soup.contents) == 1:
        attr = "section-b"
        temp = copy.copy(soup)
        split_on_tag(
            temp, "html", lambda tag: split_tag_enum(tag, is_bold, 0), attr=attr
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    for count in range(1, 6):
        attr = "section-u%d" % (count,)
        temp = copy.copy(soup)
        split_on_tag(
            temp,
            "html",
            lambda tag: split_tag_enum(tag, is_underline, count),
            attr=attr,
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    for count in range(1, 6):
        attr = "section-li-u%d" % (count,)
        temp = copy.copy(soup)
        split_on_tag(
            temp,
            "html",
            lambda tag: split_tag_list(tag, is_underline, count),
            attr=attr,
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    if len(soup.contents) == 1:
        attr = "section-u"
        temp = copy.copy(soup)
        split_on_tag(
            temp, "html", lambda tag: split_tag_enum(tag, is_underline, 0), attr=attr
        )
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    for count in range(1, 6):
        attr = "section-t%d" % (count,)
        temp = copy.copy(soup)
        split_on_tag(temp, "html", lambda tag: split_text_enum(tag, count), attr=attr)
        if len(temp.contents) >= len(soup.contents) + 5:
            soup = temp

    results = []
    prev_section_text = None

    for section in soup.find_all("html"):
        section_text = []
        for attr in ['h1', 'h2', 'h3', 'h4', 'h5',
                     'b1', 'b2', 'b3', 'b4', 'b5',
                     'li-b1', 'li-b2', 'li-b3', 'li-b4', 'li-b5',
                     'b',
                     'u1', 'u2', 'u3', 'u4', 'u5',
                     'li-u1', 'li-u2', 'li-u3', 'li-u4', 'li-u5',
                     'u',
                     't1', 't2', 't3', 't4', 't5']:
            try:
                text = section["section-%s" % (attr,)]
            except KeyError:
                continue

            text, _ = extract_index(text)
            if len(text) > 0:
                section_text.append(text)

        # If a section doesn't contain any text, and the following section isn't a subsection
        # of the previous one, then print the innermost section as regular text.
        if prev_section_text and (
            len(section_text) < len(prev_section_text)
            or section_text[: len(prev_section_text)] != prev_section_text
        ):
            results.append(
                {"text": prev_section_text[-1], "section": prev_section_text[:-1]}
            )

        prev_section_text = copy.copy(section_text)

        for paragraph in section.find_all("p"):
            text = re.sub("\\s+", " ", paragraph.get_text()).strip()
            text, _ = extract_index(text)
            if len(text) > 0:
                results.append({"text": text, "section": section_text})
                prev_section_text = None

    if prev_section_text:
        results.append(
            {"text": prev_section_text[-1], "section": prev_section_text[:-1]}
        )

    counter = collections.defaultdict(int)
    for result in results:
        counter[result["text"]] += 1

    for text, count in counter.items():
        if count > 2:
            results = [result for result in results if result["text"] != text]

    return results


def tests():
    html = "<p>This</p><pre>is\na\nsample</pre><p>text</p>"
    soup = BeautifulSoup(html, "html.parser")
    convert_pres(soup)
    assert str(soup) == "<p>This</p><p>is</p><p>a</p><p>sample</p><p>text</p>"

    html = '<span class="underline">Underline</span><span><span>More text</span></span>'
    soup = BeautifulSoup(html, "html.parser")
    convert_spans(soup)
    assert str(soup) == '<u class="underline">Underline</u>More text'

    html = "<body><p>Hello</p>World</body>"
    soup = BeautifulSoup(html, "html.parser")
    convert_strings(soup)
    assert str(soup) == "<body><p><p>Hello</p>World</p></body>"
    split_on_nested(soup, ["p"])
    assert str(soup) == "<body><p>Hello</p><p>World</p></body>"

    html = "<body>Hello<p>World</p></body>"
    soup = BeautifulSoup(html, "html.parser")
    convert_strings(soup)
    assert str(soup) == "<body><p>Hello<p>World</p></p></body>"
    split_on_nested(soup, ["p"])
    assert str(soup) == "<body><p>Hello</p><p>World</p></body>"

    html = "<p>Hello<br />World</p>"
    soup = BeautifulSoup(html, "html.parser")
    split_on_tag(soup, "p", "br")
    assert str(soup) == "<p>Hello</p><p>World</p>"

    html = "<p>Hello<p>World</p>Test</p>"
    soup = BeautifulSoup(html, "html.parser")
    split_on_nested(soup, ["p"])
    assert str(soup) == "<p>Hello</p><p>World</p><p>Test</p>"

    html = "<p>Hello<h1>World</h1>Test</p>"
    soup = BeautifulSoup(html, "html.parser")
    split_on_nested(soup, ["p", "h1"])
    assert str(soup) == "<p>Hello</p><h1>World</h1><p>Test</p>"

    html = "<p>Hello<b><u><h1>World</h1></u></b>Test</p>"
    soup = BeautifulSoup(html, "html.parser")
    split_on_nested(soup, ["p", "h1"])
    assert str(soup) == "<p>Hello</p><h1><b><u>World</u></b></h1><p>Test</p>"

    html = "<p><h1>Hello World</h1></p>"
    soup = BeautifulSoup(html, "html.parser")
    split_on_nested(soup, ["p", "h1"])
    assert str(soup) == "<h1>Hello World</h1>"

    html = "<p><div>Hello World</div></p>"
    soup = BeautifulSoup(html, "html.parser")
    delete_divs(soup)
    assert str(soup) == "<p>Hello World</p>"

    html = "<b>Hello</b><b>World</b>"
    soup = BeautifulSoup(html, "html.parser")
    merge_consecutive(soup)
    assert str(soup) == "<b>HelloWorld</b>"

    html = "<b>Hello</b> <b>World</b>"
    soup = BeautifulSoup(html, "html.parser")
    merge_consecutive(soup)
    assert str(soup) == "<b>Hello World</b>"

    html = "<b>Hello</b>&nbsp;<b>World</b>"
    soup = BeautifulSoup(html, "html.parser")
    merge_consecutive(soup)
    assert str(soup) == "<b>Hello\xa0World</b>"


if __name__ == "__main__":
    import argparse
    import json
    import os

    def check_directory(path):
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError("%s is not a valid path" % (path,))
        if not os.access(path, os.R_OK):
            raise argparse.ArgumentTypeError("%s is not a readable directory" % (path,))
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Document directory", type=check_directory)
    parser.add_argument("--lang", help="Language model", default="en")
    args = parser.parse_args()

    file_list = []
    for filename in sorted(os.listdir(args.directory)):
        if filename.endswith(".html"):
            file_list.append(os.path.join(args.directory, filename))

    def process_file(filename):
        output_filename = "%s.raw.json" % (filename[:-5],)
        print("Extracting", filename)

        with open(filename, "r") as fp:
            html = fp.read()

        if os.path.exists(output_filename):
            return filename  # Skip processing
            # os.remove(output_filename)

        if len(html.strip()) == 0:
            return filename  # Empty file

        result = extract_content(html, language=args.lang)
        if len(result) == 0:
            return filename  # No useful results, skip

        with open(output_filename, "w", encoding="utf8") as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)

        return filename

    with Pool(5) as pool:
        for i, filename in enumerate(pool.imap_unordered(process_file, file_list)):
            print("File %s processed (%f%%)." % (filename, i * 100.0 / len(file_list)))

    print("Done.")
