"""
Helper functions for other scripts
"""
import re
from functools import lru_cache

key_dict = {
    "limitation of liability": ["limitation of liability", "limitations of liability", "liability",
                                "limitation on liability", "content liability", "liability disclaimer",
                                "our liability", "limited liability", "limitations", "exclusions and limitations",
                                "disclaimer of liability", "disclaimer and limitation of liability",
                                "disclaimers and limitation of liability", "exclusion of liability",
                                "statue of limitations", "disclaimers & limitations of liability",
                                "limitations on liability", "limitation of our liability", "liability limitation",
                                "limitations and exclusions of liability", "no liability",
                                "disclaimers and limitations of liability",
                                "disclaimers; limitation of liability","limitation of liabilities"],
    "indemnification": ["indemnification", "indemnity"],
    "termination": ["termination", "termination of use", "termination of service", "term; termination",
                    "cancellation", "cancellation policy", "severance", "cancellation and termination",
                    "cancellations", "termination/access restriction", "term and termination",
                    "suspension and termination", "effect of termination", "account termination policy",
                    "termination of access", "account termination","termination and access restriction",
                    "cancellation rights"],
    "disclaimer": ["disclaimer", "disclaimers"],
    "warranty disclaimer": ["disclaimer of warranties", "no warranties", "no warranty", "warranty disclaimer",
                            "warranty", "warranties", "disclaimer of warranty", "limited warranties",
                            "disclaimer of warranties; limitation of liability", "warranty disclaimers",
                            "disclaimer of warranties and limitation of liability", "warranties and disclaimers",
                            "representations and warranties", "general representation and warranty",
                            "disclaimer of warranties and liability", "exclusion of warranties",
                            "limitation of warranties","your representations and warranties","disclaimers of warranties"],
    "law and jurisdiction": ["governing law", "governing law and jurisdiction", "applicable law", "jurisdiction",
                             "law and jurisdiction", "choice of law", "governing law & jurisdiction",
                             "applicable law and jurisdiction", "jurisdictional issues", "applicable laws",
                             "governing law; dispute resolution", "jurisdiction and applicable law",
                             "choice of law and forum", "governing law; jurisdiction", "governing law and venue",
                             "governing law and dispute resolution", "choice of law and venue", "law",
                             "applicable law and venue", "jurisdiction and venue", "choice of law and jurisdiction",
                             "governing laws","final provisions"],
    "miscellaneous": ["miscellaneous", "other", "miscellaneous provisions", "other terms",
                      "other applicable terms", "other provisions", "other important terms","additional information"],
    "severability": ["severability", "waiver and severability", "waiver & severability", "severability; waiver",
                     "severability and integration"],
    "privacy": ["privacy", "privacy policy", "your privacy", "privacy statement", "data protection", "privacy notice",
                "privacy & use of personal information", "privacy and protection of personal information"],
    "intellectual property": ["intellectual property", "intellectual property rights",
                              "compliance with intellectual property laws", "intellectual property information",
                              "intellectual and other proprietary rights","our intellectual property",
                              "intellectual property ownership"],
    "trademark": ["trademarks", "trademark information", "copyright and trademarks", "trade marks",
                  "copyright and trademark notices", "trademarks and copyrights", "copyrights and trademarks",
                  "trademark & patents", "copyrights; restrictions on use", "trademark notice","trademarks & patents"],
    "assignment": ["assignment", "waiver", "no waiver", "class action waiver", "non-waiver", "no class actions"],
    "copyright": ["copyright", "copyright policy", "copyright notice", "copyrights", "copyright infringement",
                  "copyright infringement and dmca policy", "digital millennium copyright act",
                  "copyright complaints", "claims of copyright infringement", "dmca notice", "copyright information",
                  "notice and procedure for making claims of copyright infringement",
                  "notice of copyright infringement", "notification of copyright infringement",
                  "copyright and content ownership",
                  "copyrights and copyright agents","content ownership"],
    "introduction": ["introduction", "preamble"],
    "contact": ["contact us", "contact information", "contacting us", "contact", "how to contact us", "contact details",
                "information about us", "about us", "our details", "who we are" , "who we are and how to contact us"],
    "changes": ["changes", "changes to terms of service", "changes to these terms", "changes to this agreement",
                "changes to the terms of use", "changes to terms", "modifications", "changes and amendments",
                "site terms of use modifications", "modification", "modifications to service", "changes to the terms",
                "modifications to the service and prices", "modification of these terms of use", "amendment",
                "modifications to services", "modification of terms", "modifications to terms",
                "changes to terms of use", "changes to these terms of use", "changes to terms and conditions",
                "modification of terms of use","changes to these terms and conditions","changes to the terms of service",
                "modification of these terms","changes to the terms of service","updates to terms","updates",
                "changes to the website"],
    "force majeure": ["force majeure", "events outside of our control", "events outside our control"],
    "definitions": ["definitions", "definitions and interpretation"],
    "user content": ["user content", "your content", "user generated content", "content", "user-generated content"],
    "registration": ["registration", "your registration obligations", "account registration", "user registration",
                     "registration obligations & passwords", "account creation", "registration and password",
                     "registration information"],
    "payment": ["payment", "payments", "payment terms", "fees", "fee", "fees and payment", "fees and payments", "price",
                "pricing", "prices", "price and payment", "billing", "charges", "payment methods", "fees; payment",
                "payment of fees", "terms of payment","pricing information","late payments","payment method",
                "prices and payment"],
    "license": ["license", "license and site access", "license grant", "use license", "limited license", "licenses",
                "license to use website", "your license to use the products", "grant of license", "licence",
                "licence to use website"],
    "cookies": ["cookies", "use of cookies","cookie policy","how we use cookies"],
    "notice": ["notices", "notice", "electronic notices","legal notice"],
    "security": ["security", "account security", "data security", "site security","security and password"],
    "general terms": ["general", "general conditions", "general terms", "general information", "general terms of use",
                      "general provisions", "general disclaimer", "general terms and conditions","generally"],
    "terms and conditions": ["terms", "terms and conditions", "terms & conditions", "terms of use", "term"
                             "terms of service", "acceptance of terms", "acceptance of these terms", "acceptance",
                             "acceptance of terms of use", "entire agreement", "agreement", "complete agreement",
                             "about these website terms of service", "your acceptance", "acceptance of agreement",
                             "acceptance of the terms of use", "terms and conditions of use", "agreement to terms",
                             "website terms of use", "acceptance of terms and conditions", "terms of website use",
                             "acceptance of the terms","acceptance of terms of service","user agreement",
                             "conditions of use"],
    "refunds": ["refund policy", "refunds", "refund", "money back guarantee","guarantee"],
    "eligibility": ["eligibility"],
    "prohibited use": ["prohibited uses", "no unlawful or prohibited use", "prohibited activities", "restrictions",
                       "restrictions on use", "use restrictions", "restrictions on use of materials",
                       "prohibited content", "prohibitions", "limitations on use","unauthorized use"],
    "links to other websites": ["links", "links to other web sites", "links to other websites",
                                "links to third party sites", "third-party links", "links to other sites",
                                "third party links", "links to third party websites", "third party websites",
                                "third party sites", "third party content", "third party services",
                                "third parties", "content posted on other websites", "third party rights",
                                "external links", "third-party services", "links to third-party sites",
                                "linked sites", "third-party content", "links to third-party websites",
                                "links from this website", "third-party links and resources", "third-party sites",
                                "links from our site", "third-party websites",
                                "links to third party sites/third party services", "linked websites",
                                "third party sites and information",
                                "linked websites","third party sites and information","links to external sites"],
    "disputes": ["dispute resolution", "disputes", "arbitration", "binding arbitration", "member disputes",
                 "arbitration agreement", "legal disputes", "dispute resolution; arbitration",
                 "dispute resolution and arbitration"],
    "confidentiality": ["confidentiality", "confidential information"],
    "communications": ["electronic communications", "communications", "use of communication services",
                       "communication", "electronic communication"],
    "ownership": ["ownership", "proprietary rights", "proprietary information", "our proprietary rights"],
    "reservation of rights": ["reservation of rights"],
    "acceptable use": ["acceptable use", "acceptable use policy", "use of site", "use of the site", "permitted use",
                       "use of the website", "use of content", "use of website", "use of services",
                       "use of the services", "use of the service", "use of this site", "permitted uses",
                       "your use of the site", "use of service", "use of our website","using our services"],
    "revisions and errata": ["revisions and errata", "revisions", "revision"],
    "submissions": ["submissions", "user submissions", "unsolicited submissions"],
    "accounts": ["accounts", "your account", "account", "user accounts", "user account", "account information"],
    "overview": ["overview"],
    "feedback": ["feedback", "user comments", "user comments, feedback and other submissions",
                 "your comments and concerns","reviews, comments, communications, and other content",
                 "user comments, feedback, and other submissions"],
    "personal data": ["personal information", "personal data", "user information", "user data", "customer data",
                      "your personal information"],
    "delivery": ["delivery", "shipping", "shipping policy", "delivery policy", "product delivery",
                 "shipping and delivery"],
    "variation": ["variation", "variation of terms", "variations"],
    "conduct": ["user conduct", "your conduct", "member conduct", "prohibited conduct", "conduct",
                "code of conduct", "rules of conduct","your conduct and responsible use of the digital services"],
    "services": ["services", "description of services", "description of service", "product descriptions",
                 "products or services (if applicable)", "service", "products", "product", "our services",
                 "product information", "the services", "the service", "product description"],
    "removal of links from our website": ["removal of links from our website"],
    "iframes": ["iframes"],
    "access": ["restricted access", "access", "accessing our website", "access to the site", "access and inference",
               "accessing our site", "accessibility"],
    "support": ["support","technical support","customer support","support services","customer service"],
    "hyperlinks": ["hyperlinking to our content", "hyperlinks", "linking", "linking to our site",
                   "linking to the website and social media features", "linking to the website",
                   "linking to our website", "linking to this website","links and linking","links to this website"],
    "exceptions": ["exceptions", "exclusions"],
    "violations": ["violations", "breaches of these terms and conditions","breach of terms","breach"],
    "advertisements": ["advertisements", "advertising", "advertisers", "promotions", "publicity",
                       "dealings with advertisers", "advertisments and promotions"],
    "taxes": ["taxes", "sales tax"],
    "complaints": ["complaints"],
    "accuracy": ["accuracy of billing and account information", "accuracy of materials", "accuracy of information",
                 "accuracy, completeness and timeliness of information", "reliance on information posted"],
    "errors": ["typographical errors", "errors, inaccuracies and omissions","errors, inaccuracies, and omissions"],
    "returns": ["returns", "return policy", "returns policy","returns and refunds"],
    "availability": ["availability", "availability, errors and inaccuracies", "product availability",
                     "service availability"],
    "purchases": ["purchases", "orders","ordering","making purchases"],
    "international use": ["international use", "international users", "special admonitions for international use"],
    "your responsibilities": ["your responsibility", "responsibility of website visitors", "responsibility",
                              "responsibility of contributors", "your obligations", "user obligations",
                              "obligations of the user","obligations of the visitor","responsibilities",
                              "user responsibilities"],
    "monitoring": ["monitoring"],
    "release": ["release","release of information"],
    "additional terms": ["additional terms"],
    "risk of loss": ["risk of loss"],
    "interpretation": ["interpretation"],
    "export control": ["export control", "export controls", "export","export restrictions","export compliance"],
    "subscriptions": ["automatic renewal", "subscriptions", "subscription", "membership","membership eligibility"],
    "children": ["children", "minors", "age of majority", "child protection", "minimum age","parental controls",
                 "age restrictions", "children’s privacy"],
    "attribution": ["attribution"],
    "software": ["software", "use of software", "mobile software", "mobile software from google play store",
                 "mobile software from apple’s app store", "mobile software from microsoft store"],
    "viruses": ["viruses", "viruses, hacking and other offences"],
    "domain names": ["domain names"],
    "backups": ["backups", "backups and data loss"],
    "insurance": ["insurance", "travel insurance"],
    "content standards": ["content standards", "website content", "inappropriate content"],
    "enforcement": ["enforcement", "enforceability"],
    "remedies": ["remedies", "injunctive relief", "remedy"],
    "compliance": ["compliance with laws", "legal compliance", "compliance with law", "compliance with applicable laws"]
}


def flip_dict(d):
    flipped_d = {}
    for key, val in d.items():
        for alternative in val:
            flipped_d[alternative] = key

    return flipped_d


def clean_title(title, grouped_keys):
    section_pattern = re.compile(r"section [0-9]{1,2} [-–] ")
    clean_title = title.lower().strip(":.,;'\"!?0123456789")
    clean_title = re.sub(section_pattern, "", clean_title)
    if clean_title in grouped_keys:
        return grouped_keys[clean_title]
    else:
        return None


@lru_cache(maxsize=1)
def get_section_pattern():
    return re.compile(r"section [0-9]{1,2} [-–] ")


def clean_text(text, min_length=20):
    text_pattern = get_text_pattern()
    text = re.sub(text_pattern, " ", text)
    # Order is important to make sure that empty strings would be ignored
    text = text.strip()

    if len(text) >= min_length:
        return text
    else:
        return None


@lru_cache(maxsize=1)
def get_text_pattern():
    # Replacing NULL byte, utf-8 space, and anything that could break output formatting.
    return re.compile(r"[\x00\xa0\t\n]")