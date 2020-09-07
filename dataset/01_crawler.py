"""
Module to crawl terms of service and similar texts from websites using Urllib.
The crawled *.html files are stored in the data/ directory for further processing.
"""

from bs4 import BeautifulSoup
from multiprocessing import Pool
import Levenshtein
import traceback
import numpy as np
import urllib
import time
import http
import ssl
import os


class WebsiteRetry(BaseException):
    pass


class WebsiteNotFound(BaseException):
    pass


def crawl(website, subpage=None, headless=True, lang="en"):
    """
    Download the content of a given website using Urllib.
    Optionally go to a subpage by following a link with a given text.
    """

    # FIXME: Language en?
    headers = {
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    if lang == "de":
        headers["Accept-Language"] = "de-DE,de;q=0.9"

    print("Opening website %s" % (website,))

    try:
        req = urllib.request.Request(website, headers=headers)
        with urllib.request.urlopen(req, timeout=60 * 3) as fp:
            charset = fp.info().get_content_charset()
            if charset is None:
                charset = "utf8"
            html = fp.read()
            try:
                html = html.decode(charset, errors="ignore")
            except LookupError:
                html = html.decode("utf-8", errors="ignore")
    except urllib.error.URLError:
        raise WebsiteRetry()
    except ssl.CertificateError:
        raise WebsiteRetry()
    except http.client.IncompleteRead:
        raise WebsiteRetry()
    except http.client.HTTPException:
        raise WebsiteRetry()
    except ConnectionError:
        raise WebsiteRetry()
    except UnicodeEncodeError:
        raise WebsiteRetry()
    except TimeoutError:
        raise WebsiteRetry()
    except OSError:
        raise WebsiteRetry()
    except UnicodeError:
        raise WebsiteRetry()
    except ValueError:
        raise WebsiteRetry()

    # If subpage is specified, go to the corresponding subpage
    # by matching the link texts.
    if subpage is not None:
        if not isinstance(subpage, list):
            subpage = [subpage]

        best_ratio = -np.inf
        next_link = None

        try:
            soup = BeautifulSoup(html, "html.parser")
        except NotImplementedError:
            raise WebsiteNotFound()  # cannot parse html

        for link in soup.find_all("a"):
            text = link.get_text()

            # FIXME: Is the Levenshtein distance appropriate here?
            ratio = max([Levenshtein.ratio(text.lower(), sp.lower()) for sp in subpage])
            if ratio > best_ratio:
                best_ratio = ratio
                next_link = link

        if not next_link or best_ratio < 0.75:
            raise WebsiteNotFound()

        headers["Referer"] = website
        print(next_link)

        try:
            website = urllib.parse.urljoin(website, next_link["href"])
        except KeyError:
            raise WebsiteNotFound()

        if website == headers["Referer"]:
            raise WebsiteNotFound()  # we ended up on the same page

        print("Opening website %s" % (website,))

        try:
            req = urllib.request.Request(website, headers=headers)
            with urllib.request.urlopen(req, timeout=60 * 3) as fp:
                charset = fp.info().get_content_charset()
                if charset is None:
                    charset = "utf8"
                html = fp.read()
                try:
                    html = html.decode(charset, errors="ignore")
                except LookupError:
                    html = html.decode("utf-8", errors="ignore")
        except urllib.error.URLError:
            raise WebsiteRetry()
        except ssl.CertificateError:
            raise WebsiteRetry()
        except http.client.IncompleteRead:
            raise WebsiteRetry()
        except http.client.HTTPException:
            raise WebsiteRetry()
        except ConnectionError:
            raise WebsiteRetry()
        except UnicodeEncodeError:
            raise WebsiteRetry()
        except TimeoutError:
            raise WebsiteRetry()
        except OSError:
            raise WebsiteRetry()
        except UnicodeError:
            raise WebsiteRetry()
        except ValueError:
            raise WebsiteRetry()

    return html


if __name__ == "__main__":
    from urllib.parse import urlparse, ParseResult
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="*", help="Website URL")
    parser.add_argument("--list", help="URL list", required=False)
    args = parser.parse_args()

    url_list = []
    for url in args.url:
        url_list.append(url.lower())

    if args.list is not None:
        with open(args.list) as fp:
            for line in fp:
                _, url = line.strip().split(",")
                url_list.append(url.lower())

    new_url_list = []
    seen = set()
    for url in url_list:
        if url not in seen:
            new_url_list.append(url)
            seen.add(url)
    url_list = new_url_list
    del seen

    if len(url_list) == 0:
        parser.print_help()
        exit(1)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def process_url(orig_url):
        url = orig_url.lower()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://%s" % (url,)

        subpage = [
            "Terms of Service",
            "Terms of Use",
            "Terms and Conditions",
            "Conditions of Use",
        ]

        urlp = urlparse(url)
        filename = os.path.join(data_dir, "%s.html" % urlp.netloc)

        if os.path.exists(filename):
            print("URL %s already processed." % (orig_url,))
            return orig_url, True

        urlp = ParseResult(urlp[0], "www." + urlp.netloc, *urlp[2:])
        url_list = [url, urlp.geturl()] * 3

        for i, url in enumerate(url_list):
            print("Crawling URL %s." % (url,))
            try:
                html = crawl(url, subpage, headless=True)
            except WebsiteRetry:
                traceback.print_exc()
                time.sleep(min(2 ** (i // 2), 30))
            except WebsiteNotFound:
                traceback.print_exc()
                return orig_url, False
            else:
                print("Storing html as %s." % (filename,))
                with open(filename, "w") as fp:
                    fp.write(html)
                return orig_url, True

        print("Failed to process %s." % (orig_url,))
        return orig_url, False

    if os.path.exists(os.path.join(data_dir, "blacklist.txt")):
        with open(os.path.join(data_dir, "blacklist.txt")) as fp:
            blacklist = set()
            for line in fp:
                url = line.strip()
                blacklist.add(url.lower())
            url_list = [url for url in url_list if url not in blacklist]

    with Pool(400) as pool:
        for i, (url, success) in enumerate(pool.imap_unordered(process_url, url_list)):
            print("URL %s processed (%f%%)." % (url, i * 100.0 / len(url_list)))
            if not success:
                with open(os.path.join(data_dir, "blacklist.txt"), "a") as fp:
                    fp.write("%s\n" % (url,))

    print("Done.")
