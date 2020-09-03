"""
Evals ensemble methods, based on pickled results from individual runs.

"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import segeval
from matplotlib import rc


def convert_to_masses(label):
    curr_len = 1
    masses = []
    for el in label:
        # If next section starts, increase id
        if el == 0:
            masses.append(curr_len)
            curr_len = 1
        else:
            curr_len += 1
    masses.append(curr_len)
    return tuple(masses)


def count_mistakes(label, preds):
    """

    :param labels: Ground truth label for each section
    :param preds: Predicted labels.
    :return: Number of differing predictions ("mistakes") across document
    """
    # Count the differing labels
    num_mistakes = sum(np.abs(label - preds))
    return num_mistakes


def calculate_pk(preds, labels, name=""):
    res = []
    for pred, label in zip(preds, labels):
        res.append(segeval.pk(pred, label))

    res = np.array(res)
    print(f"P_k error rate of {name} is : {np.mean(res) * 100:.2f}%")
    return np.mean(res) * 100


def load_sentence_transformers_result(fn):
    with open(fn, "r") as f:
        lines = f.readlines()

    results = []
    for line in lines:
        # output format sucks
        results.append(np.array(eval("[" + line + "]")))

    return results


def load_transformer_pickle(fp):
    with open(fp, "rb") as f:
        preds = pickle.load(f)
    return preds


def get_labels(fp="../resources/transformer_results/labels_bert_og_consec_1.pkl"):
    """
    Convenience wrapper around loading since results are kept separate.
    :param fp:
    :return: Ground truth labels of algo
    """
    return load_transformer_pickle(fp)


def get_random_baseline(labels):
    # Create random sampling baseline, knowing how many sections are in article
    all_rands = []
    for sample in labels:
        curr_paragraphs = len(sample)
        curr_sections = len(sample) - sum(sample)
        rands = np.random.choice([0, 1],
                                 len(sample),
                                 replace=True,
                                 p=[curr_sections / curr_paragraphs,
                                    1 - (curr_sections / curr_paragraphs)])
        all_rands.append(rands)
    return all_rands


def eval_model_type(same_type_preds, labels, name="all"):
    mistakes = []
    majority_preds = []
    for i, current_preds in enumerate(zip(*same_type_preds)):
        # for j, pred in enumerate(current_preds):
        #     print(names[j], len(pred))
        # print(f"Length of labels: {len(labels[i])}")
        # Simulate ensemble for now
        pred = np.stack(current_preds)

        # Majority vote. NumPy by default rounds 0.5 to 0.
        # Offset by small amount to avoid. Note this should never happen,
        # unless we have an even number of models in the ensemble.
        majority_pred = np.round(np.average(pred, axis=0) + 0.001)
        majority_preds.append(majority_pred)

        num_mistakes = count_mistakes(labels[i], majority_pred)
        # print(num_mistakes, len(labels[i]))
        mistakes.append(num_mistakes)
    # convert so binary functions work
    mistakes = np.array(mistakes)
    label_masses = [convert_to_masses(label) for label in labels]

    # Convert all other models, too
    p_ks = []
    for i, preds in enumerate(same_type_preds):
        curr_model_masses = [convert_to_masses(pred) for pred in preds]
        p_ks.append(calculate_pk(curr_model_masses, label_masses, name + "_" + str(i)))
    p_ks = np.array(p_ks)

    ensemble_masses = [convert_to_masses(pred) for pred in majority_preds]
    calculate_pk(ensemble_masses, label_masses, "ensemble-" + name)
    print(f"P_k avg error rate of {name} is : {np.mean(p_ks):.2f}% +/- "
          f"{np.std(p_ks):.2f}")

    return mistakes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the evaluation based on the ensamble of the methods, '
                    'and plots for the paper. ')
    parser.add_argument('--transformer_path',
                        help='The path to the transformer models ',
                        default="../resources/transformer_results")
    parser.add_argument('--graphseg_preds_path',
                        help='The location of the graphseg predictions ',
                        default="../resources/graphseg-results/")
    parser.add_argument('--wikiseg_preds_path',
                        help='the location of the wikiseg predictions. ',
                        default="../resources/wikiseg-results/")
    parser.add_argument('--baseline_dir',
                        help='The where the models for the baselines are located ',
                        default="../resources/baselines")
    parser.add_argument('--sentence_transformer_dir',
                        help='The folder where the models for the sentence-transformers'
                             ' are located.',
                        default="../resources/sentence-transformers")
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="../resources")

    args = parser.parse_args()

    # Set Matplotlib font to LaTeX font
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    labels = get_labels()
    transformer_path = args.transformer_path
    all_preds = []
    names = []

    # Random Baseline
    np.random.seed(321)
    random_preds = [get_random_baseline(labels) for _ in range(5)]

    graphseg_preds = []
    for i in range(1, 6):
        graphseg_fp = os.path.join(args.graphseg_preds_path, f"graphseg_results-{i}.csv")
        graphseg_preds.append(load_sentence_transformers_result(graphseg_fp))

    wikiseg_preds = []
    for i in range(1, 6):
        wikiseg_fp = os.path.join(args.wikiseg_preds_path, f"wiki_seg_results_{i}.csv")
        wikiseg_preds.append(load_sentence_transformers_result(wikiseg_fp))

    roberta_consec_preds = []
    roberta_random_preds = []
    # Load all transformer models
    for f in sorted(os.listdir(transformer_path)):
        # Don't read labels of course
        if "labels" in f:
            continue
        if "_roberta" in f:
            if "consec" in f:
                roberta_consec_preds.append(load_transformer_pickle(os.path.join(transformer_path, f)))
                # Only append roberta consec models
                all_preds.append(load_transformer_pickle(os.path.join(transformer_path, f)))
                names.append(f)
            elif "random" in f:
                roberta_random_preds.append(load_transformer_pickle(os.path.join(transformer_path, f)))

    avg_glove_preds = []
    avg_glove_random_preds = []
    tf_idf_preds = []
    tf_idf_random_preds = []
    bow_preds = []
    bow_random_preds = []
    # Check for existence of sentence-transformer baselines
    baseline_dir = args.baseline_dir
    if os.path.isdir(baseline_dir):
        for subdir, dirs, files in os.walk(baseline_dir):
            for file in files:
                # Only append the results, not labels or any other files
                if file == "prediction_results.csv":
                    full_fp = os.path.join(subdir, file)
                    if "avg_word" in subdir:
                        if "consec" in subdir:
                            avg_glove_preds.append(load_sentence_transformers_result(full_fp))
                        else:
                            avg_glove_random_preds.append(load_sentence_transformers_result(full_fp))
                    elif "tf-idf" in subdir:
                        if "consec" in subdir:
                            tf_idf_preds.append(load_sentence_transformers_result(full_fp))
                        else:
                            tf_idf_random_preds.append(load_sentence_transformers_result(full_fp))
                    elif "bow" in subdir:
                        if "consec" in subdir:
                            bow_preds.append(load_sentence_transformers_result(full_fp))
                        else:
                            bow_random_preds.append(load_sentence_transformers_result(full_fp))
                    # all_preds.append(load_sentence_transformers_result(full_fp))
                    # names.append(subdir)

    base_preds = []
    base_random_preds = []
    nli_preds = []
    nli_random_preds = []
    # Check for existence of sentence-transformer baselines
    st_dir = args.sentence_transformer_dir
    if os.path.isdir(st_dir):
        for subdir, dirs, files in os.walk(st_dir):
            for file in files:
                # Only append the results, not labels or any other files
                if file == "prediction_results.csv":
                    full_fp = os.path.join(subdir, file)
                    if "roberta-base-nli" in subdir:
                        if "og_consec" in subdir:
                            nli_preds.append(load_sentence_transformers_result(full_fp))
                            all_preds.append(load_sentence_transformers_result(full_fp))
                            names.append(subdir)
                        else:
                            nli_random_preds.append(load_sentence_transformers_result(full_fp))
                    else:
                        if "og_consec" in subdir:
                            base_preds.append(load_sentence_transformers_result(full_fp))
                            all_preds.append(load_sentence_transformers_result(full_fp))
                            names.append(subdir)
                        else:
                            base_random_preds.append(load_sentence_transformers_result(full_fp))

    num_sections = 0
    num_paragraphs = 0
    mistakes = []
    majority_preds = []
    for i, current_sample_preds in enumerate(zip(*all_preds)):
        num_paragraphs += len(labels[i])
        num_sections += len(labels[i]) - sum(labels[i])
        # for j, pred in enumerate(current_sample_preds):
        #     print(names[j], len(pred))
        # print(f"Length of labels: {len(labels[i])}")

        # Simulate ensemble for now
        pred = np.stack(current_sample_preds)

        # Majority vote. NumPy by default rounds 0.5 to 0.
        # Offset by small amount to avoid. Note this should never happen,
        # unless we have an even number of models in the ensemble.
        majority_pred = np.round(np.average(pred, axis=0) + 0.001)
        majority_preds.append(majority_pred)

        num_mistakes = count_mistakes(labels[i], majority_pred)
        # print(num_mistakes, len(labels[i]))
        mistakes.append(num_mistakes)

    print(f"Average number of sections: {num_sections / len(labels):.2f}.")  # 6.27
    print(f"Average length: {num_paragraphs / len(labels):.2f} paragraphs")  # 22.14

    # Ensemble results
    label_masses = [convert_to_masses(label) for label in labels]
    ensemble_masses = [convert_to_masses(pred) for pred in majority_preds]
    calculate_pk(ensemble_masses, label_masses, "Ensemble")

    # convert so binary functions work
    mistakes = np.array(mistakes)

    mistakes_random = eval_model_type(random_preds, labels, name="Random Oracle")
    mistakes_graphseg = eval_model_type(graphseg_preds, labels, name="GraphSeg")
    mistakes_wikiseg = eval_model_type(wikiseg_preds, labels, name="WikiSeg")
    # CLS runs
    mistakes_roberta_consec = eval_model_type(roberta_consec_preds, labels, name="RoBERTa CLS Consec")
    mistakes_roberta_random = eval_model_type(roberta_random_preds, labels, name="RoBERTa CLS Random")

    # Baseline runs
    mistakes_avg_glove = eval_model_type(avg_glove_preds, labels, name="Avg GloVe Consec")
    mistakes_avg_glove_random = eval_model_type(avg_glove_random_preds, labels, name="Avg GloVe Random")
    mistakes_tf_idf = eval_model_type(tf_idf_preds, labels, name="TF-IDF Consec")
    mistakes_tf_idf_random = eval_model_type(tf_idf_random_preds, labels, name="TF-IDF Random")
    mistakes_bow = eval_model_type(bow_preds, labels, name="BoW Consec")
    mistakes_bow_random = eval_model_type(bow_random_preds, labels, name="BoW Random")

    # ST runs
    mistakes_st_base = eval_model_type(base_preds, labels, name="ST roberta-base Consec")
    mistakes_st_base_random = eval_model_type(base_random_preds, labels, name="ST roberta-base Random")
    mistakes_st_nli = eval_model_type(nli_preds, labels, name="ST roberta-nli-mean Consec")
    mistakes_st_nli_random = eval_model_type(nli_random_preds, labels, name="ST roberta-nli-mean Random")

    # # Convert all other models, too
    # for i, preds in enumerate(all_preds):
    #     curr_model_masses = [convert_to_masses(pred) for pred in preds]
    #     calculate_pk(curr_model_masses, label_masses, names[i])

    # Plotting of results versus mistakes
    x = list(range(10))
    y = []
    y_random_baseline = []
    y_graphseg = []
    y_wikiseg = []
    y_roberta_consec = []
    y_roberta_random = []
    y_avg_glove = []
    y_avg_glove_random = []
    y_tf_idf = []
    y_tf_idf_random = []
    y_bow = []
    y_bow_random = []
    y_st_base = []
    y_st_base_random = []
    y_st_nli = []
    y_st_nli_random = []

    for i in x:
        y.append(sum(mistakes <= i) / len(mistakes))
        y_random_baseline.append(sum(mistakes_random <= i) / len(mistakes))
        y_graphseg.append(sum(mistakes_graphseg <= i) / len(mistakes))
        y_wikiseg.append(sum(mistakes_wikiseg <= i) / len(mistakes))
        y_roberta_consec.append(sum(mistakes_roberta_consec <= i) / len(mistakes))
        y_roberta_random.append(sum(mistakes_roberta_random <= i) / len(mistakes))
        y_avg_glove.append(sum(mistakes_avg_glove <= i) / len(mistakes))
        y_avg_glove_random.append(sum(mistakes_avg_glove_random <= i) / len(mistakes))
        y_tf_idf.append(sum(mistakes_tf_idf <= i) / len(mistakes))
        y_tf_idf_random.append(sum(mistakes_tf_idf_random <= i) / len(mistakes))
        y_bow.append(sum(mistakes_bow <= i) / len(mistakes))
        y_bow_random.append(sum(mistakes_bow_random <= i) / len(mistakes))
        y_st_base.append(sum(mistakes_st_base <= i) / len(mistakes))
        y_st_base_random.append(sum(mistakes_st_base_random <= i) / len(mistakes))
        y_st_nli.append(sum(mistakes_st_nli <= i) / len(mistakes))
        y_st_nli_random.append(sum(mistakes_st_nli_random <= i) / len(mistakes))

    plt.xticks(x)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlim([0, 9])
    plt.ylim([0, 1])
    plt.xlabel(r"Number of mistakes $k$")
    plt.ylabel(r"$acc_{k}$")

    plt.plot(x, y, marker="o", label="Ens All", color="#a65628", linestyle="--")
    plt.plot(x, y_random_baseline, marker="x", label="Rand Oracle",
             color="#a65628", linestyle="--")
    plt.plot(x, y_avg_glove, marker="v", label=r"GLV$avg$ CP",
             color="#e41a1c", linestyle="-")
    plt.plot(x, y_tf_idf, marker="^", label="tf-idf CP",
             color="#377eb8", linestyle="-")
    plt.plot(x, y_bow, marker="<", label="BoW CP",
             color="#4daf4a", linestyle="-")
    plt.plot(x, y_roberta_consec, marker="s", label="Ro-CLS CP",
             color="#984ea3", linestyle="-")
    plt.plot(x, y_st_base, marker="P", label="ST-Ro CP",
             color="#ff7f00", linestyle="-")
    plt.plot(x, y_st_nli, marker="X", label="ST-Ro-N CP",
             color="#ffff33", linestyle="-")
    plt.plot(x, y_avg_glove_random, fillstyle="none", marker="v", label=r"GLV$avg$ RP",
             color="#e41a1c", linestyle=":")
    plt.plot(x, y_tf_idf_random, fillstyle="none", marker="^", label="tf-idf RP",
             color="#377eb8", linestyle=":")
    plt.plot(x, y_bow_random, fillstyle="none", marker="<", label="BoW RP",
             color="#4daf4a", linestyle=":")
    plt.plot(x, y_roberta_random, fillstyle="none", marker="s", label="Ro-CLS RP",
             color="#984ea3", linestyle=":")
    plt.plot(x, y_st_base_random, fillstyle="none", marker="P", label="ST-Ro RP",
             color="#ff7f00", linestyle=":")
    plt.plot(x, y_st_nli_random, fillstyle="none", marker="X", label="ST-Ro-N RP",
             color="#ffff33", linestyle=":")

    l3 = plt.legend(bbox_to_anchor=(0.47, 1.25), loc="upper center", borderaxespad=0, ncol=4)
    plt.savefig(os.path.join(args.output_folder, "p_k.png"),
                dpi=300, bbox_extra_artists=(l3,), bbox_inches='tight')
    print(y)
