import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Merge, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from keras import optimizers
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from keras.layers import Input, Embedding, LSTM, Convolution1D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import MaxPooling1D, AveragePooling1D, Bidirectional
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from sklearn.externals import joblib
import gensim, logging
import multiprocessing
import random
from keras.utils import np_utils, generic_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import merge, Dropout, Flatten, Dense, Permute
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import pysam
from collections import defaultdict
import os
import argparse
import timeit
import re
import pyBigWig
import tempfile
import sys
import hashlib
from multiprocessing import Process
import keras
from sklearn import metrics
from gensim.models.word2vec import LineSentence
import gensim, logging
import pickle
import numpy as np
import keras
from keras.layers import Dense, LSTM, Dropout, Bidirectional
import gensim, logging
from keras.models import load_model
from keras.layers import Concatenate

def suffle_text(file_input, file_output):
    f = open(file_input)
    oo = open(file_output, 'w')
    entire_file = f.read()
    file_list = entire_file.split('\n')
    num_lines = len(file_list)
    random_nums = random.sample(xrange(num_lines), num_lines)
    for i in random_nums:
        oo.write(file_list[i] + "\n")

    oo.close()
    f.close()


def seq2ngram(seqs, k, s, dest, wv):
    f = open(seqs)
    lines = f.readlines()
    f.close()
    list22 = []
    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')

    for num, line in enumerate(lines):
        if num < 200000:
            line = line[:-1].lower()  # remove '\n' and lower ACGT
            l = len(line)  # length of line
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
            list22.append(convert_data_to_index(list2, wv))
    f.close()
    return list22


def convert_sequences_to_index(list_of_seqiences, wv):
    ll = []
    for i in range(len(list_of_seqiences)):
        ll.append(convert_data_to_index(list_of_seqiences[i], wv))
    return ll


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def seq2ngram2(seqs, k, s, dest):
    f = open(seqs)
    lines = f.readlines()
    f.close()

    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')
    for num, line in enumerate(lines):
        if num < 100000:
            line = line[:-1].lower()  # remove '\n' and lower ACGT
            l = len(line)  # length of line

            for i in range(0, l, s):
                if i + k >= l + 1:
                    break

                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
    f.close()


def word2vect(k, s, vector_dim, root_path, pos_sequences):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    seq2ngram2(pos_sequences, k, s, 'seq_pos_' + str(k) + '_' + str(s) + '.txt')
    sentences = LineSentence('seq_pos_' + str(k) + '_' + str(s) + '.txt')

    mode1 = gensim.models.Word2Vec(sentences, iter=20, window=int(18 / s), min_count=50, size=vector_dim,
                                   workers=multiprocessing.cpu_count())
    mode1.save(root_path + 'word2vec_model' + '_' + str(k) + '_' + str(s) + '_' + str(vector_dim))


def build_class_file(np, ng, class_file):
    with open(class_file, 'w') as outfile:
        outfile.write('label' + '\n')
        for i in range(np):
            outfile.write('1' + '\n')
        for i in range(ng):
            outfile.write('0' + '\n')


def build_ACNN_BLSTM_model(k, s, vector_dim, root_path, MAX_LEN, pos_sequences, neg_sequences, seq_file, class_file,model_dir):
    model1 = gensim.models.Word2Vec.load(
        model_dir + 'word2vec_model' + '_' + str(k) + '_' + str(s) + '_' + str(vector_dim))

    pos_list = seq2ngram(pos_sequences, k, s, 'seq_pos_' + str(k) + '_' + str(s) + '.txt', model1.wv)
    with open(str(k) + '_' + str(s) + 'listpos.pkl', 'wb') as pickle_file:
        pickle.dump(pos_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(str(k) + '_' + str(s) + 'listpos.pkl', 'rb') as f:
    #    pos_list = pickle.load(f)
    # pos_list = pos_list[:250]
    # print(str(len(pos_list)))

    neg_list = seq2ngram(neg_sequences, k, s, 'seq_neg_' + str(k) + '_' + str(s) + '.txt', model1.wv)
    with open(str(k) + '_' + str(s) + 'listneg.pkl', 'wb') as pickle_file:
        pickle.dump(neg_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(str(k) + '_' + str(s) + 'listneg.pkl', 'rb') as f1:
    #    neg_list = pickle.load(f1)
    # neg_list = neg_list[:200]
    # print (str(len(neg_list)))
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN)
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    build_class_file(len(pos_list), len(neg_list), class_file)
    X1 = X

    n_seqs = len(seqs)
    indices = np.arange(n_seqs)

    np.random.shuffle(indices)

    X = X[indices]

    y = y[indices]

    n_tr = int(n_seqs * 0.8)

    X_train = X[:n_tr]

    y_train = y[:n_tr]

    X_valid = X[n_tr:]

    y_valid = y[n_tr:]

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],

                        output_dim=embedding_matrix.shape[1],

                        weights=[embedding_matrix],

                        input_length=MAX_LEN,

                        trainable=True))
    model.add(Dropout(0.1))

    # model.add(Convolution1D(nb_filter = 100,filter_length=1,activation='relu',border_mode = 'valid'))


    model.add(Convolution1D(nb_filter=100,

                            filter_length=7,

                            activation='relu',

                            border_mode='valid'))

    model.add(MaxPooling1D(4, 4))

    model.add(Dropout(0.1))

    # model.add(Convolution1D(nb_filter = 80,filter_length=1,activation='relu',border_mode = 'valid'))

    model.add(Convolution1D(100, 1, activation='relu'))

    model.add(MaxPooling1D(2, 2))

    model.add(Dropout(0.1))

    model.add(Bidirectional(LSTM(100, consume_less='gpu')))

    model.add(Dropout(0.1))
    model.add(Dense(80, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(20, activation='relu', name='myfeatures'))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    print(model.summary())

    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # print(model.summary())


    checkpointer = ModelCheckpoint(
        filepath=model_dir+'bestmodel_ACNN_BLSTM_' + str(k) + ' ' + str(s) + ' ' + str(vector_dim) + str(MAX_LEN) + '.hdf5',
        verbose=1,
        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print('Training model...')
    history = model.fit(X_train, y_train, nb_epoch=2, batch_size=128, shuffle=True,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer, earlystopper],
                        verbose=1)
    # print(history.history.keys())
    # summarize history for accuracy
    # plt.figure()
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('C:/Users/mohamed/Documents/text_mining/finaldata/myaccuracy-drop='+str(int(ls*10))+'s='+str(s)+'vectrdim='+str(vector_dim))
    # summarize history for loss
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('C:/Users/mohamed/Documents/text_mining/finaldata/myloss-drop='+str(int(ls*10))+'s='+str(s)+'vectrdim='+str(vector_dim))
    # tresults = model.evaluate(X_test, y_test)
    # print (tresults)
    # y_pred = model.predict(X_test, batch_size=32, verbose=1)
    # y = y_test
    # print ('Calculating AUC...')
    # auroc = metrics.roc_auc_score(y, y_pred)
    # auprc = metrics.average_precision_score(y, y_pred)
    # print (auroc, auprc)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('myfeatures').output)
    np.savetxt(seq_file, intermediate_layer_model.predict(X1), delimiter=" ")


def extract_ACNN_BLSTM(k, s, vector_dim, root_path, MAX_LEN, testing_sequences, seq_file,model_dir):
    model1 = gensim.models.Word2Vec.load(
        model_dir + 'word2vec_model' + '_' + str(k) + '_' + str(s) + '_' + str(vector_dim))

    seqs = seq2ngram(testing_sequences, k, s, 'seq_' + str(k) + '_' + str(s) + '.txt', model1.wv)

    X = pad_sequences(seqs, maxlen=MAX_LEN)
    model = load_model(model_dir+'bestmodel_ACNN_BLSTM_' + str(k) + ' ' + str(s) + ' ' + str(vector_dim) + str(MAX_LEN) + '.hdf5')
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('myfeatures').output)
    np.savetxt(seq_file, intermediate_layer_model.predict(X), delimiter=" ")


def bigwig_score_list(bw, chr, start, end):
    score = []
    kk = bw.intervals(chr, start, end)
    if kk != None:
        for t in bw.intervals(chr, start, end):
            score.append(t[2])

    if len(score) == 0:
        for i in range(start, end):
            score.append(0)
    return score


def bigwig_mean(bw, chr, start, end):
    score_sum = 0
    mean_score = 0
    kk = bw.intervals(chr, start, end)
    if kk != None:

        for t in bw.intervals(chr, start, end):
            score_sum += t[2]
    else:
        print('yes')

    if (end - start) != 0:

        mean_score = score_sum / (end - start)

    else:
        mean_score = 0

    return mean_score


def extract_exons(gtf_file):
    gtf = open(gtf_file, 'r');
    exons = defaultdict(list)
    for line in gtf:  ## process each line
        if line[0] != '#':
            ele = line.strip().split('\t');
            if len(ele) > 7:

                if ele[2] == 'exon':
                    chr = (ele[0])
                    strand = (ele[6])
                    start = int(ele[3])
                    end = int(ele[4])

                    exons[chr + strand].append([start, end])

    return exons


def get_processed_conservation_score(score_whole_seq, thres):
    ls = len(score_whole_seq)

    score_array = np.array(score_whole_seq)

    con_arr = (score_array >= thres)
    con_str = ''
    for val in con_arr:
        if val:
            con_str = con_str + '1'
        else:
            con_str = con_str + '0'
    sat8_len = con_str.count('11111111')
    sat7_len = con_str.count('1111111')
    sat_6len = con_str.count('111111')
    sat5_len = con_str.count('11111')

    return float(sat5_len) * 1000 / ls, float(sat_6len) * 1000 / ls, float(sat7_len) * 1000 / ls, float(
        sat8_len) * 1000 / ls


def point_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))


def extract_feature_conservation_CCF(fasta_file, bigwig, gtf_file, out):
    fp = open(fasta_file, 'r')
    bw = pyBigWig.open(bigwig)
    exons = extract_exons(gtf_file)

    fw = open(out, 'w')
    ii = 0
    for line in fp:
        ii = ii + 1
        ele = line.strip().split('	')

        chr_name = ele[0]
        start = int(ele[1])
        end = int(ele[2]) - 1
        strand = ele[3]
        list_all_exons = exons[chr_name + strand]
        list_exons = []

        score = []
        tt = True
        for i in range(len(list_all_exons)):
            start_exon = list_all_exons[i][0]
            end_exon = list_all_exons[i][1]
            if point_overlap(start_exon, end_exon, start, end):
                for i in range(len(list_exons)):
                    if list_exons[i][0] == start_exon and list_exons[i][1] == end_exon:
                        tt = False
                if tt:
                    list_exons.append((start_exon, end_exon))

        b = []
        for begin, end in sorted(list_exons):
            if b and b[-1][1] >= begin - 1:
                b[-1][1] = max(b[-1][1], end)
            else:
                b.append([begin, end])
        list_exons = b
        if len(list_exons) == 0:
            list_exons.append([start, end])

        for i in range(len(list_exons)):
            score.append(bigwig_mean(bw, chr_name, list_exons[i][0], list_exons[i][1]))
        score_array = np.array(score)
        mean_score = score_array.mean()
        max_score = score_array.max()
        median_score = np.median(score_array)
        fw.write(str(mean_score) + ' ' + str(max_score) + ' ' + str(
            median_score))
        score_whole_seq = bigwig_score_list(bw, chr_name, start, end)

        l5, l6, l7, l8 = get_processed_conservation_score(score_whole_seq, 0.5)

        fw.write(' ' + str(l5) + ' ' + str(l6) + ' ' + str(l7) + ' ' + str(l8))
        l5, l6, l7, l8 = get_processed_conservation_score(score_whole_seq, 0.6)

        fw.write(' ' + str(l5) + ' ' + str(l6) + ' ' + str(l7) + ' ' + str(l8))
        l5, l6, l7, l8 = get_processed_conservation_score(score_whole_seq, 0.7)

        fw.write(' ' + str(l5) + ' ' + str(l6) + ' ' + str(l7) + ' ' + str(l8))
        l5, l6, l7, l8 = get_processed_conservation_score(score_whole_seq, 0.8)

        fw.write(' ' + str(l5) + ' ' + str(l6) + ' ' + str(l7) + ' ' + str(l8))
        l5, l6, l7, l8 = get_processed_conservation_score(score_whole_seq, 0.9)

        fw.write(' ' + str(l5) + ' ' + str(l6) + ' ' + str(l7) + ' ' + str(l8))
        fw.write('\n')

    fp.close()
    bw.close()


def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complseq = [complement[base] for base in seq]
    return complseq


def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


def bed_to_fasta(bed_file, fasta_file, sequence_file, genome_fasta):
    bed = open(bed_file, 'r');

    output = open(fasta_file, 'w');
    output2 = open(sequence_file, 'w');
    genome_fa = pysam.FastaFile(genome_fasta)
    for line in bed:

        values = line.split()

        chr_n = values[0]
        start = int(values[1])
        end = int(values[2])
        strand = values[3]
        seq = genome_fa.fetch(chr_n, start, end)

        seq = seq.upper()

        if strand == '-':
            seq = reverse_complement(seq)
        output.write('>' + ':'.join([chr_n, str(start), str(end), strand]) + '\n')
        output.write(seq + '\n')
        output2.write(seq + '\n')


def extract_rcm(fasta_file, genome, out, kk, jj):
    genome_fa = pysam.FastaFile(genome)
    fp = open(fasta_file, 'r')
    fw = open(out, 'w')
    indices, number_kmers = get_dictionary_kmers(jj)
    numline = 0
    for line in fp:
        ele = line.strip().split(':')
        if line[0] == '>':
            numline = numline + 1
            if numline % 10 == 0:
                print('%d lines made' % numline)
            chr_name = ele[0][1:]
            start = int(ele[1])
            end = int(ele[2]) - 1
            strand = ele[3]
            scores = get_rcm_score(chr_name, strand, start, end, genome_fa, kk, jj, indices, number_kmers)
            for jjj in range(len(scores)):
                fw.write(str(scores[jjj]) + ' ')
            fw.write('\n')


def extract_rcm2(fasta_file, genome, out, kk, jj):
    genome_fa = pysam.FastaFile(genome)
    fp = open(fasta_file, 'r')
    fw = open(out, 'w')
    indices, number_kmers = get_dictionary_kmers(jj)
    numline = 0
    for line in fp:
        ele = line.strip().split(':')
        if line[0] == '>':
            numline = numline + 1
            if numline % 10 == 0:
                print('%d lines made' % numline)
            chr_name = ele[0][1:]
            start = int(ele[1])
            end = int(ele[2]) - 1
            strand = ele[3]
            scores = get_rcm_score2(chr_name, strand, start, end, genome_fa, kk, jj, indices, number_kmers)
            for jjj in range(len(scores)):
                fw.write(str(scores[jjj]) + ' ')
            fw.write('\n')


def match_score(alpha, beta):
    match_award = 12
    mismatch_penalty = -2
    if alpha == beta:

        return match_award
    else:
        return mismatch_penalty


def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval


def water(seq1, seq2):
    m, n = len(seq1) - 1, len(seq2) - 1  # length of two sequences

    # Generate DP table and traceback path pointer matrix
    score = zeros((m + 1, n + 1))  # the DP table
    pointer = zeros((m + 1, n + 1))  # to store the traceback path
    max_score_500 = 0
    max_score_750 = 0
    max_score_1000 = 0
    max_score_1250 = 0

    # initial maximum score in DP table
    # Calculate DP table and mark pointers

    gap_penalty = -12

    for i in range(1, min(m + 1, 1250)):
        for j in range(1, min(n + 1, 1250)):
            score_diagonal = score[i - 1][j - 1] + match_score(seq1[i], seq2[j]);

            score[i][j] = max(0, score_diagonal)

            if score[i][j] >= max_score_500 and i <= 500 and j <= 500:
                max_score_500 = score[i][j];
            if score[i][j] >= max_score_750 and i <= 750 and j <= 750:
                max_score_750 = score[i][j];
            if score[i][j] >= max_score_1000 and i <= 1000 and j <= 1000:
                max_score_1000 = score[i][j];
            if score[i][j] >= max_score_1250 and i <= 1250 and j <= 1250:
                max_score_1250 = score[i][j];

    return [max_score_500, max_score_750, max_score_1000, max_score_1250]


def get_rcm_score(chrr, strand, start, end, genome_fa, wsize, motifsize, indices, number_kmers):
    fi_seq = genome_fa.fetch(chrr, start - wsize, start).upper()
    si_seq = genome_fa.fetch(chrr, end, end + wsize).upper()

    if strand == '-':
        fi_seq = reverse_complement(fi_seq)
        si_seq = reverse_complement(si_seq)

    fi_seq_ind = sequence_to_indices(fi_seq, indices, motifsize)
    fsi_seq_ind = sequence_to_indices_rc(si_seq, indices, motifsize)

    results = water(fi_seq_ind, fsi_seq_ind)

    return results


def get_rcm_score2(chrr, strand, start, end, genome_fa, wsize, motifsize, indices, number_kmers):
    fi_seq = genome_fa.fetch(chrr, start - wsize, start).upper()
    si_seq = genome_fa.fetch(chrr, end, end + wsize).upper()

    if strand == '-':
        fi_seq = reverse_complement(fi_seq)
        si_seq = reverse_complement(si_seq)

    fi_seq_ind = sequence_to_indices(fi_seq, indices, motifsize)
    fsi_seq_ind = sequence_to_indices_rc(si_seq, indices, motifsize)

    results2 = []
    for size in [500, 750, 1000, 1250, 1500, 1750, 1990]:
        results2.append(absolute_number_rcm(fi_seq_ind[:size], fsi_seq_ind[2000 - size:], number_kmers))
        # ss=extract_absolute_number_rcm(fi_seq_ind,fsi_seq_ind):
    return results2


def absolute_number_rcm(fi_seq_ind, fsi_seq_ind, end):
    sum = 0
    for i in range(end):
        sum = sum + min(fi_seq_ind.count(i), fsi_seq_ind.count(i))
    return sum


def sequence_to_indices(sequence, indices, k):
    seq_list = []
    for i in range(len(sequence) - k):
        seq_list.append(indices[sequence[i:i + k]])
    return seq_list


def sequence_to_indices_rc(sequence, indices, k):
    seq_list = []
    for i in range(len(sequence) - k):
        seq_list.append(indices[reverse_complement(sequence[i:i + k])])
    return seq_list


def get_dictionary_kmers(k):
    indices = defaultdict(int)

    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** k
    for i in range(0, end):
        seq = ''
        n = i
        for j in range(k):
            seq = seq + chars[n % base]
            n = int(n / base)
        indices[seq] = i
    return indices, end


def extract_rcm_features(pos_data_fasta, neg_data_fasta, genome, rcm_file, data_dir):
    if not os.path.exists(data_dir + 'rcm/'):
        os.makedirs(data_dir + 'rcm/')
    proc = []
    for k in [1, 2, 3]:
        out = data_dir + 'rcm/pos_rcm1_' + str(k)
        print(str(k))
        p = Process(target=extract_rcm, args=(pos_data_fasta, genome, out, 1250, k))
        proc.append(p)
        # extract_rcm(pos_data_fasta, genome, out, 1000, 3)
    for p in proc:
        p.start()

    proc2 = []
    for k in [1, 2, 3]:
        out = data_dir + 'rcm/neg_rcm1_' + str(k)
        print(str(k))
        p2 = Process(target=extract_rcm, args=(neg_data_fasta, genome, out, 1250, k))
        proc2.append(p2)
        # extract_rcm(pos_data_fasta, genome, out, 1000, k)
    for p2 in proc2:
        p2.start()

    proc3 = []
    for k in [3, 4, 5, 6]:
        out = data_dir + 'rcm/pos_rcm2_' + str(k)
        print(str(k))
        p3 = Process(target=extract_rcm2, args=(pos_data_fasta, genome, out, 2000, k))
        proc3.append(p3)
        # extract_rcm(pos_data_fasta, genome, out, 1000, 3)
    for p3 in proc3:
        p3.start()

    proc4 = []
    for k in [3, 4, 5, 6]:
        out = data_dir + 'rcm/neg_rcm2_' + str(k)
        print(str(k))
        p4 = Process(target=extract_rcm2, args=(neg_data_fasta, genome, out, 2000, k))
        proc4.append(p4)
        # extract_rcm(pos_data_fasta, genome, out, 1000, k)
    for p4 in proc4:
        p4.start()

    for p in proc:
        p.join()

    for p2 in proc2:
        p2.join()

    for p3 in proc3:
        p3.join()

    for p4 in proc4:
        p4.join()
    concatenate_rcm_files(rcm_file, data_dir)


def concatenate_rcm_files(rcm_file, data_dir):
    file_dir = data_dir + 'rcm/'
    root_range = [1, 2]
    K_range = {1: [1, 2, 3],
               2: [3, 4, 5, 6]}

    # dict to store the data as the files a read out
    lines = {'pos': list(), 'neg': list(), 'header': ""}

    def append_data(filename, column_prefix, sign):
        """
        take each line from file, and append that line to
        respective string in lines[sign]
        """
        with open(os.path.join(file_dir, filename)) as f:
            # initialize the lines list if first time
            new_lines = f.readlines()
            if len(lines[sign]) == 0:
                lines[sign] = ["" for _ in range(len(new_lines))]

            # append every line to lines
            last_line = str()
            for i, (line, new_data) in enumerate(zip(lines[sign], new_lines)):
                lines[sign][i] += new_data.strip() + ' '
                last_line = new_data.strip()

            # append headers
            if sign == 'pos':
                for i in range(len(last_line.split(' '))):
                    lines['header'] += column_prefix + '_' + str(i + 1) + ' '

    # interate through all file names and call append_data on each
    for root in root_range:
        for K in K_range[root]:
            for sign in ['pos', 'neg']:
                column_prefix = str(root) + '_' + str(K)
                filename = sign + '_rcm' + column_prefix
                print(filename)
                append_data(filename, column_prefix, sign)

    # write to outfile.txt in pwd
    with open(rcm_file, 'w') as outfile:
        outfile.write(lines['header'] +  '\n')
        for line in lines['pos']:
            outfile.write(line + '\n')
        for line in lines['neg']:
            outfile.write(line + '\n')


def extract_rcm_features_testing(data_fasta, genome, rcm_file, data_dir):
    if not os.path.exists(data_dir + 'rcm/'):
        os.makedirs(data_dir + 'rcm/')
    proc = []
    for k in [1, 2, 3]:
        out = data_dir + 'rcm/rcm1_' + str(k)
        p = Process(target=extract_rcm, args=(data_fasta, genome, out, 1250, k))
        proc.append(p)
        # extract_rcm(pos_data_fasta, genome, out, 1000, 3)
    for p in proc:
        p.start()

    proc3 = []
    for k in [3, 4, 5, 6]:
        out = data_dir + 'rcm/rcm2_' + str(k)
        print(str(k))
        p3 = Process(target=extract_rcm2, args=(data_fasta, genome, out, 2000, k))
        proc3.append(p3)
        # extract_rcm(pos_data_fasta, genome, out, 1000, 3)
    for p3 in proc3:
        p3.start()

    for p in proc:
        p.join()

    for p3 in proc3:
        p3.join()
    print('start concatenate')
    concatenate_rcm_files_testing(rcm_file, data_dir)


def concatenate_rcm_files_testing(rcm_file, data_dir):
    file_dir = data_dir + 'rcm/'
    root_range = [1, 2]
    K_range = {1: [1, 2, 3],
               2: [3, 4, 5, 6]}

    # dict to store the data as the files a read out
    lines = {'data': list(), 'header': ""}

    def append_data(filename, column_prefix, sign='data'):
        """
        take each line from file, and append that line to
        respective string in lines[sign]
        """
        with open(os.path.join(file_dir, filename)) as f:
            # initialize the lines list if first time
            new_lines = f.readlines()
            if len(lines[sign]) == 0:
                lines[sign] = ["" for _ in range(len(new_lines))]

            # append every line to lines
            last_line = str()
            for i, (line, new_data) in enumerate(zip(lines[sign], new_lines)):
                lines[sign][i] += new_data.strip() + ' '
                last_line = new_data.strip()

            # append headers
            if sign == 'data':
                for i in range(len(last_line.split(' '))):
                    lines['header'] += column_prefix + '_' + str(i + 1) + ' '

    # interate through all file names and call append_data on each
    for root in root_range:
        for K in K_range[root]:
            column_prefix = str(root) + '_' + str(K)
            filename = 'rcm' + column_prefix
            print(filename)
            append_data(filename, column_prefix, 'data')

    # write to outfile.txt in pwd
    with open(rcm_file, 'w') as outfile:
        outfile.write(lines['header'] + '\n')
        for line in lines['data']:
            outfile.write(line + '\n')


def concatenate_cons_files(pos_conservation_feature_file, neg_conservation_feature_file, cons_file):
    filenames = [pos_conservation_feature_file, neg_conservation_feature_file]
    with open(cons_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())


def extract_features_testing(testing_bed, genome, bigwig, gtf, data_dir,model_dir):
    cons_file = data_dir + 'conservation_features_test.txt'
    rcm_file = data_dir + 'rcm_features_test.txt'
    seq_file = data_dir + 'seq_features_test.txt'

    testing_fasta = testing_bed + '.fasta'
    testing_sequences = testing_bed + '.seq.txt'

    bed_to_fasta(testing_bed,testing_fasta ,testing_sequences, genome)

    extract_feature_conservation_CCF(testing_bed, bigwig, gtf, cons_file)

    extract_rcm_features_testing(testing_fasta, genome,rcm_file,data_dir)
    extract_ACNN_BLSTM(3, 1, 40, data_dir, 8000, testing_sequences, seq_file,model_dir)

    return testing_fasta, testing_sequences, cons_file, rcm_file, seq_file


def extract_features_training(pos_data_bed, neg_data_bed, genome, bigwig, gtf, data_dir,model_dir):
    cons_file = data_dir + 'conservation_features.txt'
    rcm_file = data_dir + 'rcm_features.txt'
    seq_file = data_dir + 'seq_features.txt'
    class_file = data_dir + 'class.txt'
    pos_data_fasta = pos_data_bed + '.fasta'
    pos_sequences = pos_data_bed + '.seq.txt'

    neg_data_fasta = neg_data_bed + '.fasta'
    neg_sequences = neg_data_bed + '.seq.txt'

    pos_conservation_feature_file = pos_data_bed + '.cons'
    neg_conservation_feature_file = neg_data_bed + '.cons'

    bed_to_fasta(pos_data_bed,pos_data_fasta ,pos_sequences, genome)
    bed_to_fasta(neg_data_bed, neg_data_fasta,neg_sequences, genome)

    extract_feature_conservation_CCF(neg_data_bed, bigwig, gtf, neg_conservation_feature_file)
    extract_feature_conservation_CCF(pos_data_bed, bigwig,gtf,pos_conservation_feature_file)
    concatenate_cons_files(pos_conservation_feature_file,neg_conservation_feature_file,cons_file)

    extract_rcm_features(pos_data_fasta, neg_data_fasta, genome,rcm_file,data_dir)

    word2vect(3, 1, 40, model_dir, pos_sequences)
    print('11111')
    build_ACNN_BLSTM_model(3, 1, 40, data_dir, 8000, pos_sequences, neg_sequences, seq_file, class_file,model_dir)
    print('22222')
    return pos_data_fasta, pos_sequences, neg_data_fasta, neg_sequences, cons_file, rcm_file, seq_file


def load_data(path, seq=True, rcm=True, cons=False, test=False, cons_file=None, rcm_file=None, seq_file=None):
    """

        Load data matrices from the specified folder.

    """

    data = dict()

    if seq: data["seq"] = np.loadtxt(seq_file, delimiter=' ', skiprows=0)

    if rcm: data["rcm"] = np.loadtxt(rcm_file, skiprows=1)

    if cons: data["cons"] = np.loadtxt(cons_file, skiprows=0)

    if test:

        data["Y"] = []

    else:

        data["Y"] = np.loadtxt(path + 'class.txt', skiprows=1)

    print('data loaded')

    return data


def split_training_validation(classes, validation_size=0.2, shuffle=False):
    """split sampels based on balnace classes"""

    num_samples = len(classes)

    classes = np.array(classes)

    classes_unique = np.unique(classes)

    num_classes = len(classes_unique)

    indices = np.arange(num_samples)

    # indices_folds=np.zeros([num_samples],dtype=int)

    training_indice = []

    training_label = []

    validation_indice = []

    validation_label = []

    print(str(classes_unique))
    for cl in classes_unique:

        indices_cl = indices[classes == cl]

        num_samples_cl = len(indices_cl)

        # split this class into k parts

        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual

        num_samples_validation = int(num_samples_cl * validation_size)
        res = num_samples_cl - num_samples_validation
        training_indice = training_indice + [val for val in indices_cl[num_samples_validation:]]
        training_label = training_label + [cl] * res
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_validation]]
        validation_label = validation_label + [cl] * num_samples_validation

    training_index = np.arange(len(training_label))

    random.shuffle(training_index)

    training_indice = np.array(training_indice)[training_index]

    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))

    random.shuffle(validation_index)

    validation_indice = np.array(validation_indice)[validation_index]

    validation_label = np.array(validation_label)[validation_index]
    print(np.shape(training_indice))
    print(np.shape(training_label))
    print(np.shape(validation_indice))
    print(np.shape(validation_label))

    return training_indice, training_label, validation_indice, validation_label


def preprocess_data(X, scaler=None, stand=False):
    if not scaler:

        if stand:

            scaler = StandardScaler()

        else:

            scaler = MinMaxScaler()

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()

        encoder.fit(labels)

    y = encoder.transform(labels).astype(np.int32)

    if categorical:
        y = np_utils.to_categorical(y)

    return y, encoder


def get_rnn_fea(train, sec_num_hidden=128, num_hidden=128):
    model = Sequential()

    # model.add(Dense(num_hidden, input_dim=train.shape[1], activation='relu'))

    model.add(Dense(num_hidden, input_shape=(train.shape[1],), activation='relu'))

    model.add(PReLU())

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(num_hidden, input_dim=num_hidden, activation='relu'))

    # model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))

    model.add(PReLU())

    model.add(BatchNormalization())

    # model.add(Activation('relu'))

    model.add(Dropout(0.5))

    return model


def train_circDeep(data_dir, model_dir, genome, bigwig, gtf, positive_bed, negative_bed, seq=True, rcm=True, cons=True,extract_features=True):
    if extract_features:
        pos_data_fasta, pos_sequences, neg_data_fasta, neg_sequences, cons_file, rcm_file, seq_file = extract_features_training(
            positive_bed, negative_bed, genome, bigwig, gtf, data_dir,model_dir)
    else:
        cons_file = data_dir + 'conservation_features.txt'
        rcm_file = data_dir + 'rcm_features.txt'
        seq_file = data_dir + 'seq_features.txt'
    training_data = load_data(data_dir, seq, rcm, cons, False, cons_file, rcm_file, seq_file)
    print('training', len(training_data))

    seq_hid = 20

    rcm_hid = 256

    cons_hid = 64

    training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data["Y"])
    print('split done')
    if seq:
        print(np.shape(training_data["seq"]))

        seq_data, seq_scaler = preprocess_data(training_data["seq"])
        print(np.shape(seq_data))
        joblib.dump(seq_scaler, os.path.join(model_dir, 'seq_scaler.pkl'))

        seq_train = seq_data[training_indice]

        seq_validation = seq_data[validation_indice]

        seq_net = get_rnn_fea(seq_train, sec_num_hidden=seq_hid, num_hidden=seq_hid * 2)

        seq_data = []

        training_data["seq"] = []

    if rcm:
        rcm_data, rcm_scaler = preprocess_data(training_data["rcm"])

        joblib.dump(rcm_scaler, os.path.join(model_dir, 'rcm_scaler.pkl'))

        rcm_train = rcm_data[training_indice]

        rcm_validation = rcm_data[validation_indice]

        rcm_net = get_rnn_fea(rcm_train, sec_num_hidden=rcm_hid, num_hidden=rcm_hid * 3)

        rcm_data = []

        training_data["rcm"] = []
    if cons:
        cons_data, cons_scaler = preprocess_data(training_data["cons"])

        joblib.dump(cons_scaler, os.path.join(model_dir, 'cons_scaler.pkl'))

        cons_train = cons_data[training_indice]

        cons_validation = cons_data[validation_indice]

        cons_net = get_rnn_fea(cons_train, sec_num_hidden=cons_hid, num_hidden=cons_hid * 3)

        cons_data = []

        training_data["cons"] = []

    y, encoder = preprocess_labels(training_label)

    val_y, encoder = preprocess_labels(validation_label, encoder=encoder)
    training_data.clear()

    model = Sequential()

    training_net = []

    training = []

    validation = []
    total_hid = 0

    if seq:
        training_net.append(seq_net)

        training.append(seq_train)

        validation.append(seq_validation)

        total_hid = total_hid + seq_hid

        seq_train = []

        seq_validation = []
    if rcm:
        training_net.append(rcm_net)

        training.append(rcm_train)

        validation.append(rcm_validation)

        total_hid = total_hid + rcm_hid

        rcm_train = []

        rcm_validation = []
    if cons:
        training_net.append(cons_net)

        training.append(cons_train)

        validation.append(cons_validation)

        total_hid = total_hid + cons_hid

        cons_train = []

        cons_validation = []
    #model.add(concatenate(training_net))
    model.add(Merge(training_net, mode='concat'))

    model.add(Dropout(0.2))

    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    print(model.summary())

    # checkpointer = ModelCheckpoint(filepath='bestmodel_circDeep.hdf5', verbose=1, save_best_only=True)

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print('model training')

    model.fit(training, y, batch_size=128, nb_epoch=15, verbose=1, validation_data=(validation, val_y),
              callbacks=[earlystopper])

    model.save(os.path.join(model_dir, 'bestmodel_circDeep.pkl'))

    #joblib.dump(model, os.path.join(model_dir,'bestmodel_circDeep.pkl'))


    return model


def test_circDeep(data_dir, model_dir, genome, bigwig, gtf, testing_bed, seq=True, rcm=True, cons=True,
                  outfile='prediction.txt',model1=None):
    testing_fasta, testing_sequences, cons_file, rcm_file, seq_file = extract_features_testing(testing_bed, genome,
                                                                                               bigwig, gtf, data_dir,model_dir)
    test_data = load_data(data_dir, seq, rcm, cons, True, cons_file, rcm_file, seq_file)

    # true_y = test_data["Y"].copy()

    testing = []

    if seq:
        seq_data, seq_scaler = preprocess_data(test_data["seq"])
        testing.append(seq_data)

    if rcm:
        rcm_data, rcm_scaler = preprocess_data(test_data["rcm"])

        testing.append(rcm_data)
    if cons:
        cons_data, cons_scaler = preprocess_data(test_data["cons"])

        testing.append(cons_data)
    if model1==None:
        try:
            # sometimes load model getting errors from many users, it is because of
            # different versions of keras and tensorflow librarues so ze recommende
            # to use last versions. To overcome this problem we save features of our training data and ze train
            # our model for few seconds each time ze wanna make test for bed file
            model1 = load_model(os.path.join(model_dir, 'bestmodel_circDeep.pkl'))
        except:
            model1 = train_circDeep(data_dir, model_dir, genome, bigwig, gtf, None, None, seq, rcm,
                                   cons, False)

    # model = joblib.load(os.path.join(model_dir, 'bestmodel_circDeep.pkl'))
    # model = joblib.load( os.path.join(model_dir,'model.pkl'))
    predictions = model1.predict_proba(testing)
    # pdb.set_trace()
    # auc = roc_auc_score(true_y, predictions[:, 1])
    # print "Test AUC: ", auc
    # fw.write(str(auc) + '\n')
    # mylabel = "\t".join(map(str, true_y))
    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, predictions[:, 1]))
    # fw.write(mylabel + '\n')
    fw.write(myprob)
    fw.close()


def run_circDeep(parser):
    data_dir = parser.data_dir

    out_file = parser.out_file

    train = parser.train

    model_dir = parser.model_dir

    predict = parser.predict

    seq = parser.seq

    rcm = parser.rcm
    genome = parser.genome
    bigwig = parser.bigwig
    gtf = parser.gtf
    positive_bed = args.positive_bed
    negative_bed = args.negative_bed
    cons = parser.cons
    testing_bed = args.testing_bed
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if predict:
        train = False

    if train:

        print('model training')

        model=train_circDeep(data_dir, model_dir, genome, bigwig, gtf, positive_bed, negative_bed, seq, rcm,
                       cons,True)
        test_circDeep(data_dir, model_dir, genome, bigwig, gtf, testing_bed, seq, rcm, cons,out_file,model)

    else:

        print('model prediction')
        model=None
        test_circDeep(data_dir, model_dir, genome, bigwig, gtf, testing_bed, seq, rcm, cons,
                      out_file,model)


def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default='data/', metavar='<data_directory>',
                        help='Under this directory, you will have descriptors files ')

    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')

    parser.add_argument('--model_dir', type=str, default='models/',
                        help='The directory to save the trained models for future prediction')

    parser.add_argument('--predict', type=bool, default=False,
                        help='Predicting circular RNAs. if using train, then it will be False')

    parser.add_argument('--out_file', type=str, default='prediction.txt',
                        help='The output file used to store the prediction probability of testing data')

    parser.add_argument('--seq', type=bool, default=True, help='The modularity of ACNN-BLSTM seq')

    parser.add_argument('--rcm', type=bool, default=True, help='The modularity of RCM')

    parser.add_argument('--cons', type=bool, default=True, help='The modularity of conservation')

    parser.add_argument('--genome', type=str, default='data/hg38.fasta', help='The Fasta file of genome')

    parser.add_argument('--gtf', type=str, default='data/Homo_sapiens.Ensembl.GRCh38.82.gtf',
                        help='The gtf annotation file. e.g., hg38.gtf')

    parser.add_argument('--bigwig', type=str, default='data/hg38.phyloP20way.bw',
                        help='conservation scores in bigWig file format')

    parser.add_argument('--positive_bed', type=str, default='data/circRNA_dataset.bed',
                        help='BED input file for circular RNAs for training, it should be like:chromosome    start    end    gene')

    parser.add_argument('--negative_bed', type=str, default='data/negative_dataset.bed',
                        help='BED input file for other long non coding RNAs for training, it should be like:chromosome    start    end    gene')
    parser.add_argument('--testing_bed', type=str, default='data/test.bed',
                        help='BED input file for testing data, it should be like:chromosome    start    end    gene')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='circular RNA classification from other long non-coding RNA using multimodal deep learning')

    args = parse_arguments(parser)

    run_circDeep(args)
