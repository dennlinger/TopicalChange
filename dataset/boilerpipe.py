import threading
import jpype
import os

if not jpype.isJVMStarted():
    jars = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boilerpipe-1.2.0")
    for top, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jar"):
                jars.append(os.path.join(top, file))
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "-Djava.class.path=%s" % os.pathsep.join(jars),
        convertStrings=True,
    )

lock = threading.Lock()

InputSource = jpype.JClass("org.xml.sax.InputSource")
StringReader = jpype.JClass("java.io.StringReader")
HTMLHighlighter = jpype.JClass("de.l3s.boilerpipe.sax.HTMLHighlighter")
BoilerpipeSAXInput = jpype.JClass("de.l3s.boilerpipe.sax.BoilerpipeSAXInput")

"""
Supported extractors:
- DefaultExtractor
- ArticleExtractor
- ArticleSentencesExtractor
- KeepEverythingExtractor
- KeepEverythingWithMinKWordsExtractor
- LargestContentExtractor
- NumWordsRulesExtractor
- CanolaExtractor
"""


def extract(html, extractor="ArticleExtractor", **kwargs):
    with lock:
        if not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()

        if extractor == "KeepEverythingWithMinKWordsExtractor":
            kMin = kwargs.get("kMin", 1)  # set default to 1
            extractor = jpype.JClass("de.l3s.boilerpipe.extractors." + extractor)(kMin)
        else:
            extractor = jpype.JClass(
                "de.l3s.boilerpipe.extractors." + extractor
            ).INSTANCE

        reader = StringReader(html)
        source = BoilerpipeSAXInput(InputSource(reader)).getTextDocument()
        extractor.process(source)

        highlighter = HTMLHighlighter.newExtractingInstance()
        return highlighter.process(source, html)


if __name__ == "__main__":
    with open("amazon.html") as fp:
        html = fp.read()

    html = extract(html)

    with open("amazon-extracted.html", "w") as fp:
        fp.write(html)
