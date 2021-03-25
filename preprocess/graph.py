from preprocess import argmanager as arg
from preprocess import sample as samp
from preprocess import analyzer as alz
from preprocess import methods as meth
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import re
import os
import pickle
import numpy as np
from scipy.optimize import curve_fit
from collections import namedtuple

results_dir = alz.results_directory

marker_list = ['.', 'o', 'v', '>', '1', 's', '*', 'd', 'P', '+']
line_style_list = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

training_sizes = [5000, 10000, 20000, 40000, 60000, 80000, 100000]

PlottingData = namedtuple('PlottingData', 'x y label')
BoxData = namedtuple('BoxData', 'x label')
TriPlotData = namedtuple('TriPlotData', 'x y yerr label')

def create_method_string(methods):
    if not methods:
        return 'No Treatment'

    return '-'.join(methods)

def create_sample_plots(corpus, methods):
    corpus_progressive = pull_corpus_progressive_samples(methods)
    method_progressive = pull_method_progressive_samples(corpus, methods)
    method_comparative = pull_method_comparative_samples(corpus, methods)

    method_string = create_method_string(methods)

    plot_sampling_data(corpus_progressive, f'{method_string} Heaps Curve Across Corpora')
    plot_sampling_data(method_progressive, f'{corpus} {method_string} Progressive')
    plot_sampling_data(method_comparative, f'{corpus} {method_string} Comparative')

def plot_sampling_data(data, title):

    plt.title(title)
    plt.xlabel('Corpus Size')
    plt.ylabel('Vocabulary Size')

    for i, datum in enumerate(data):
        plt.plot(datum.x, datum.y, marker=marker_list[i], linestyle=line_style_list[i], color=f'C{i}', label=datum.label)

    plt.legend()
    plot_title = title.replace(' ', '_')
    plt.savefig(f'{plot_title}.png')
    plt.clf()

def retrieve_type_token_from_sample(sample):
    type_index = samp.TYPE_INDEX
    token_index = samp.TOKEN_INDEX

    x = [s[token_index] for s in sample][::10000]
    y = [s[type_index] for s in sample][::10000]

    return x, y

def pull_method_comparative_samples(corpus, methods):
    samples = list()

    # Need vanilla baseline
    vanilla_methods = list()
    x, y = retrieve_type_token_from_sample(get_sample(corpus, vanilla_methods))
    samples.append(PlottingData(x, y, create_method_string(vanilla_methods)))

    for method in methods:
        sample = get_sample(corpus, [method])
        if sample is None: continue

        x, y = retrieve_type_token_from_sample(sample)
        label = method

        samples.append(PlottingData(x, y, label))

    return samples

def pull_corpus_progressive_samples(methods):

    samples = list()

    for corpus in arg.valid_corpora:
        sample = get_sample(corpus, methods)
        if sample is None: continue

        x, y = retrieve_type_token_from_sample(sample)
        label = corpus

        samples.append(PlottingData(x, y, label))

    return samples

def pull_method_progressive_samples(corpus, methods):

    samples = list()

    # Need vanilla baseline
    vanilla_methods = list()
    x, y = retrieve_type_token_from_sample(get_sample(corpus, vanilla_methods))
    samples.append(PlottingData(x, y, create_method_string(vanilla_methods)))

    progressive_methods = list()

    for method in methods:
        progressive_methods.append(method)

        sample = get_sample(corpus, progressive_methods)
        if sample is None: continue

        x, y = retrieve_type_token_from_sample(sample)
        label = create_method_string(progressive_methods)

        samples.append(PlottingData(x, y, label))

    return samples

def get_sample(corpus, methods):
    #TODO: Will samples always be 1d arrays? Does hashing make a nd array?
    sample_file = samp.create_sample_filename(corpus, methods)

    if not os.path.isfile(sample_file): return None

    with open(sample_file, 'rb') as f:
        sample = pickle.load(f)

    return sample

def calculate_heaps_parameters(x, y):
    def heaps_curve(x, k, beta):
        return k * (x ** beta)

    params, params_covariance = curve_fit(heaps_curve, x, y, bounds=([-np.inf, 0], [np.inf, .999999]))

    return params[0], params[1]

def plot_results(corpus, methods, model, attributes=['accuracy', 'traintime', 'vocabsize'], mode='comparative'):
    fig, axes = plt.subplots(len(attributes), 1, sharex=True, figsize=(10, 10))

    method_string = create_method_string(methods)

    title = f'{corpus} {method_string} {mode} Results'
    axes[0].set_title(f'{title} Results')
    axes[-1].set_xlabel('Corpus Size')

    for i, attribute in enumerate(attributes):
        results = pull_size_progressive_results(corpus, methods, model, attribute, mode)

        for j, result in enumerate(results):
            axes[i].errorbar(result.x, result.y, yerr=result.yerr, label=result.label, marker=marker_list[j], linestyle=line_style_list[j], color=f'C{j}')
            axes[i].set_ylabel(label_dict[attribute])

    plt.legend()
    save_title = title.replace(' ', '_')
    plt.savefig(f'{save_title}_results.png')
    plt.clf()

def create_result_box_plots(corpus, methods, model, attribute='traintime', mode='comparative'):

    fig, axes = plt.subplots(len(training_sizes), 1, sharex=True, figsize=(10, 10))
    method_string = create_method_string(methods)

    title = f'{corpus} {method_string} {mode} {label_dict[attribute]}'
    axes[0].set_title(f'{title} Results')
    axes[-1].set_xlabel('Treatment')

    for i, train_size in enumerate(sorted(training_sizes, reverse=True)):
        results = mode_dict[mode](corpus, methods, model, train_size, attribute)

        X = [r.x for r in results]
        labels = [r.label for r in results]

        axes[i].set_ylabel(f'{train_size}')

        if i == (len(training_sizes) - 1):
            axes[i].boxplot(X, labels=labels)
        else:
            axes[i].boxplot(X)

    save_title = title.replace(' ', '_')
    plt.savefig(f'{save_title}_box.png')
    plt.clf()

def box_plot_results(data, title, attribute, train_size=10000):

    label_dict = {
                    'traintime' : 'Train Time',
                    'accuracy'  : 'Accuracy',
                 }

    plt.title(title)
    plt.xlabel('Treatment')
    plt.ylabel(f'{label_dict[attribute]}\n({train_size})')

    X = [d.x for d in data]
    labels = [d.label for d in data]

    plt.boxplot(X, labels=labels)
    save_title = title.replace(' ', '_')
    plt.savefig(f'{save_title}_box.png')
    plt.clf()

def pull_method_progressive_results(corpus, methods, model, train_size, attribute):

    results = list()
    progressive_methods = list()

    vanilla_result = get_result(corpus, list(), model, train_size, attribute)
    results.append(BoxData(vanilla_result, 'No Treatment'))

    for method in methods:
        progressive_methods.append(method)
        result = get_result(corpus, progressive_methods, model, train_size, attribute)

        if result is None: continue

        label = create_method_string(progressive_methods)

        results.append(BoxData(result, label))

    return results

def pull_method_comparative_results(corpus, methods, model, train_size, attribute):

    results = list()

    vanilla_result = get_result(corpus, list(), model, train_size, attribute)
    results.append(BoxData(vanilla_result, 'No Treatment'))

    for method in methods:
        result = get_result(corpus, [method], model, train_size, attribute)

        if result is None: continue

        results.append(BoxData(result, method))

    return results

def pull_size_progressive_results(corpus, methods, model, attribute, mode='comparative'):

    results_puller = mode_dict[mode]

    y_axis = list()
    labels = list()
    x_axis = list()

    for size in training_sizes:
        y_data = results_puller(corpus, methods, model, size, attribute)
        x_data = results_puller(corpus, methods, model, size, 'tokens')

        for i, datum in enumerate(y_data):
            if datum.label not in labels: labels.append(datum.label)
            if i >= len(y_axis):
                y_axis.append(list())
                x_axis.append(list())

            y_axis[i].append((np.mean(datum.x), np.std(datum.x)))
            x_axis[i].append(np.mean(x_data[i].x))

    results = list()

    for i, y_data in enumerate(y_axis):
        y = [d[0] for d in y_axis[i]]
        yerr = [d[1] for d in y_axis[i]]
        x = x_axis[i]

        results.append(TriPlotData(x, y, yerr, labels[i]))

    return results

def get_data_from_result(result, attribute):
    return [getattr(r, attribute) for r in result]

def get_result(corpus, methods, model, train_size, attribute, seed=0):

    seed_replace = re.compile(r'_0.pickle')
    results_file = seed_replace.sub('.pickle', alz.generate_analyze_filename(corpus, methods, model, train_size, seed))

    if not os.path.isfile(results_file): return None

    with open(results_file, 'rb') as f:
        result = pickle.load(f)

    return get_data_from_result(result, attribute)

label_dict = {
                   'traintime' : 'Train Time',
                    'accuracy' : 'Accuracy',
                  'vocabsize' : 'Vocabulary Size',
                      'tokens' : 'Corpus Size',
             }

mode_dict = {
                'progressive' : pull_method_progressive_results,
                'comparative' : pull_method_comparative_results,
            }

if __name__ == '__main__':
    args = arg.parse_args()

    if args.model:
        create_result_box_plots(args.corpus, args.methods, args.model)
        plot_results(args.corpus, args.methods, args.model)
