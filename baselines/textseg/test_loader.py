from torch.utils.data import Dataset
from baselines.textseg import word_model
from baselines.textseg import extract_sentence_words
from pathlib2 import Path
import re
import baselines.textseg.wiki_utils as wiki_utils

section_delimiter = "========"



def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if high_granularity else wiki_utils.get_seperator_foramt(
        (1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')


    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.decode('utf-8').strip()

    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]

    return sections

def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section

def read_file(path, word2vec, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data = []
    targets = []
    all_sections = get_sections(path, high_granularity)

    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if 1 <= len(sentence_words):
                        data.append([word_model(word, word2vec) for word in sentence_words])
                    else:
                        #raise ValueError('Sentence in wikipedia file is empty')
                        logger.info('Sentence in wikipedia file is empty')
                else:  # for the annotation. keep sentence as is.
                    if (only_letters):
                        sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                        data.append(sentence)
                    else:
                        data.append(sentence)
            if data:
                targets.append(len(data) - 1)

    return data, targets, path

def get_files(path):
    all_objects = Path(path).glob('*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

class TestDataSet(Dataset):
    def __init__(self, root, word2vec, train=False, manifesto=False, folder=False, high_granularity=False):


        root_path = Path(root)
        self.textfiles =get_files(root_path)

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                              high_granularity=self.high_granularity)

    def __len__(self):
        return len(self.textfiles)