# coding=utf-8

from scipy.io import savemat
from joint_bayesian import *
from sklearn.preprocessing import normalize
from sklearn.lda import LDA
import os
import get_feature
from sklearn.externals import joblib


def excute_train(train_data, train_label, result_fold='../result/'):
    """
    train pca and plda then save the models to result_fold
    """
    data = load_mat_directly(train_data) * 1.0
    # data = data.transpose()
    data = normalize(data, norm='l2')
    # data = data_pre(data)
    label = load_mat_directly(train_label)

    pca = PCA_Train(data, result_fold, n_components=128)
    data_pca = pca.transform(data)
    data_pca = substract_mean(data_pca)
    save_pac_mat(data_pca, train_data)
    JointBayesian_Train(data_pca, label, result_fold)
    # lda = LDA()
    # lda.fit(data_pca, label)
    # joblib.dump(lda, result_fold + "lda_model.m")


def get_A_G_testset_feature(test_data, result_fold='../result/'):
    """
    get matrix A,G and feature of lfw dataset
    :return: A,G is para in plda,data[i] is ith image's feature in lfw dataset
    """
    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)

    pca_file = get_pca_filename(test_data)
    if not os.path.exists(pca_file):
        data = load_mat_directly(test_data) * 1.0
        data = normalize(data, norm='l2')
        # data = data_pre(data)
        data = pca_transform(result_fold + 'pca_model.m', data)
        save_pac_mat(data, test_data)

    data = load_mat_directly(pca_file)
    return A, G, data


def excute_test(pairlist, test_data, result_fold='../result/'):
    pair_list = load_mat_directly(pairlist)
    test_Intra = pair_list['IntraPersonPair'][0][0] - 1
    test_Extra = pair_list['ExtraPersonPair'][0][0] - 1

    A, G, data = get_A_G_testset_feature(test_data, result_fold)
    # lda = joblib.load(result_fold + 'lda_model.m')
    # data = lda.transform(data)

    dist_Intra = get_ratios(A, G, test_Intra, data)
    dist_Extra = get_ratios(A, G, test_Extra, data)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))
    for i in range(0, len(dist_all), 100):
        print(dist_all[i], label[i])
    data_to_pkl({'distance': dist_all, 'label': label}, result_fold + 'result.pkl')


def verification(person1, person2, threshold, result_fold='../result/'):
    """
    compare between two images and report whether they are the same person
    :param person1: first person's name,it should include in lfw dataset,such as Aaron_Eckhart_0001
    :param person2: same as person1
    :param threshold: threshold to judge whether they are the same person
    """
    '''
    if not os.path.exists(result_fold + 'image_name_to_id.pkl'):
        imagelist_lfw = load_mat_directly('../data/imagelist_lfw.mat')
        image_name_to_id = {}
        for i, image_name in enumerate(imagelist_lfw):
            image_name_to_id[(str(image_name[0][0])).split('\\')[1].split('.')[0]] = i
        data_to_pkl(image_name_to_id, result_fold + 'image_name_to_id.pkl')
    image_name_to_id = read_pkl(result_fold + 'image_name_to_id.pkl')
    if image_name_to_id.has_key(person1) and image_name_to_id.has_key(person2):
        A, G, data = get_A_G_testset_feature(test_data, result_fold)
    '''
    person1 = name_to_path(person1)
    person2 = name_to_path(person2)
    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)
    fea = get_feature.get_feature([person1, person2])
    fea = pca_transform(result_fold + 'pca_model.m', fea)
    distance = Verify(A, G, fea[0], fea[1])
    if distance > threshold:
        print 'they are same person'
    else:
        print 'they are not same person'


if __name__ == '__main__':
    '''
    data1 = read_pkl('../result/lfw_fea.pkl')
    data2 = load_mat_directly('/home/liuxuebo/CV/BLUFR/code/test_lfwa/pubdata/lfw.mat')['wfea']
    data2=data2.transpose()
    for i in range(len(data1)):
        print(sum(data1[i] - data2[i]))
    '''
    excute_train(train_data='../data/googlenet_lfw.mat', train_label='../data/id_lfw.mat')
    excute_test(pairlist='../data/pairlist_lfw.mat', test_data='../data/googlenet_lfw.mat')
    excute_performance("../result/result.pkl", -50, -40, 0.1)
    while True:
        person1 = raw_input('please input two images\' name, if you want to quit, input exit\n')
        if person1 == 'exit':
            break
        person2 = raw_input()
        verification(person1, person2, threshold=-47)
