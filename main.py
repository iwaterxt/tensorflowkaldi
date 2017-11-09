'''@file main.py
run this file to go through the neural net training procedure, look at the config files in the config directory to modify the settings'''

import os
import numpy as np
from six.moves import configparser
from neuralNetworks import nnet
from processing import ark, prepare_data, feature_reader, batchdispenser, target_coder, ark2hdf5
from kaldi import gmm


#here you can set which steps should be executed. If a step has been executed in the past the result have been saved and the step does not have to be executed again (if nothing has changed)
GMMTRAINFEATURES = False 	#required
GMMDEVFEATURES = False       #required
GMMTESTFEATURES = False	 	#required if the performance of a GMM is tested
DNNTRAINFEATURES = False	    #required
DNNDEVFEATURES = False       #required
DNNTESTFEATURES = False	 	#required if the performance of the DNN is tested
TRAIN_MONO = False 			#required
ALIGN_MONO = False			#required
TEST_MONO = False 			#required if the performance of the monphone GMM is tested
TRAIN_TRI = False			#required if the triphone or LDA GMM is used for alignments
ALIGN_TRI = False			#required if the triphone or LDA GMM is used for alignments
TEST_TRI = False			    #required if the performance of the triphone GMM is tested
TRAIN_LDA = False			#required if the LDA GMM is used for alignments
ALIGN_LDA = False			#required if the LDA GMM is used for alignments
TEST_LDA = False			    #required if the performance of the LDA GMM is tested
DNNFEATURESPRO = False     #required if the features need global transform
DNNTESTFEATPRO = False       #required if the test features need global transform
TRAINARKTOHDF5 = True
DEVARKTOHDF5 = True
TRAIN_NNET = True			#required
TEST_NNET = True			#required if the performance of the DNN is tested


#read config file
config = configparser.ConfigParser()
config.read('config/config_thchs30.cfg')
current_dir = os.getcwd()

os.system('''source path.sh''')

#compute the features of the training set for GMM training
if GMMTRAINFEATURES:
    feat_cfg = dict(config.items('gmm-features'))

    print '------- computing GMM training features ----------'
    prepare_data.prepare_data(config.get('directories', 'train_data'), config.get('directories', 'train_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])

if GMMDEVFEATURES:
    feat_cfg = dict(config.items('gmm-features'))

    print '------- computing GMM dev features --------------'
    prepare_data.prepare_data(config.get('directories', 'dev_data'), config.get('directories', 'dev_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(config.get('directories', 'dev_features') + '/' + feat_cfg['name'])

#compute the features of the training set for DNN training if they are different then the GMM features
if DNNTRAINFEATURES:
    if config.get('dnn-features', 'name') != config.get('gmm-features', 'name'):
        feat_cfg = dict(config.items('dnn-features'))

        print '------- computing DNN training features ----------'
        prepare_data.prepare_data(config.get('directories', 'train_data'), config.get('directories', 'train_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print '------- computing cmvn stats ----------'
        prepare_data.compute_cmvn(config.get('directories', 'train_features') + '/' + feat_cfg['name'])

if DNNDEVFEATURES:
    if config.get('dnn-features', 'name') != config.get('gmm-features', 'name'):
        feat_cfg = dict(config.items('dnn-features'))

        print '------- computing DNN dev features ----------'
        prepare_data.prepare_data(config.get('directories', 'dev_data'), config.get('directories', 'dev_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print '------- computing cmvn stats ----------'
        prepare_data.compute_cmvn(config.get('directories', 'dev_features') + '/' + feat_cfg['name'])
#compute the features of the training set for GMM testing
if GMMTESTFEATURES:
    feat_cfg = dict(config.items('gmm-features'))

    print '------- computing GMM testing features ----------'
    prepare_data.prepare_data(config.get('directories', 'test_data'), config.get('directories', 'test_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

    print '------- computing cmvn stats ----------'
    prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])

#compute the features of the training set for DNN testing if they are different then the GMM features
if DNNTESTFEATURES:
    if config.get('dnn-features', 'name') != config.get('gmm-features', 'name'):
        feat_cfg = dict(config.items('dnn-features'))

        print '------- computing DNN testing features ----------'
        prepare_data.prepare_data(config.get('directories', 'test_data'), config.get('directories', 'test_features') + '/' + feat_cfg['name'], feat_cfg, feat_cfg['type'], feat_cfg['dynamic'])

        print '------- computing cmvn stats ----------'
        prepare_data.compute_cmvn(config.get('directories', 'test_features') + '/' + feat_cfg['name'])


#use kaldi to train the monophone GMM
mono_gmm = gmm.MonoGmm(config)
if TRAIN_MONO:
    mono_gmm.train()

#get alignments with the monophone GMM
if ALIGN_MONO:
    mono_gmm.align()

#use kaldi to test the monophone GMM
if TEST_MONO:
    mono_gmm.test()

#use kaldi to train the triphone GMM
tri_gmm = gmm.TriGmm(config)
if TRAIN_TRI:
    tri_gmm.train()

#get alignments with the triphone GMM
if ALIGN_TRI:
    tri_gmm.align()

#use kaldi to test the triphone gmm
if TEST_TRI:
    tri_gmm.test()

#use kaldi to train the LDA+MLLT GMM
lda_gmm = gmm.LdaGmm(config)
if TRAIN_LDA:
    lda_gmm.train()

#get alignments with the LDA+MLLT GMM
if ALIGN_LDA:
    lda_gmm.align()

#use kaldi to test the LDA+MLLT gmm
if TEST_LDA:
    lda_gmm.test()


#get the feature input dim
reader = ark.ArkReader(config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name') + '/feats.scp')
_, features, _ = reader.read_next_utt()
input_dim = features.shape[1]

#get number of output labels
numpdfs = open(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph/num_pdfs')
num_labels = numpdfs.read()
num_labels = int(num_labels[0:len(num_labels)-1])
numpdfs.close()


splice = int(config.get('nnet', 'splice'))
splice_step = int(config.get('nnet', 'splice_step'))

splice_vec = range(-splice*splice_step, splice*splice_step+1, splice_step)
input_dim = len(splice_vec)*input_dim
#create the neural net
nnet = nnet.Nnet(config, input_dim, num_labels)

if TRAIN_NNET:

    #only shuffle if we start with initialisation
    if config.get('nnet', 'starting_step') == '0':
        #shuffle the examples on disk
        print '------- shuffling examples ----------'
        prepare_data.shuffle_examples(config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'))

    #put all the alignments in one file
    alifiles = [config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '_ali/pdf.' + str(i+1) + '.gz' for i in range(int(config.get('general', 'num_jobs')))]
    alifile = config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '_ali/pdf.all'
    os.system('cat %s > %s' % (' '.join(alifiles), alifile))

    #put all the dev data alignments in one file
    alifiles_dev = [config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '_dev_ali' + '/pdf.' + str(i+1) + '.gz' for i in range(int(config.get('general', 'num_jobs')))]
    alifile_dev = config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '_dev_ali' + '/pdf.all'
    os.system('cat %s > %s' % (' '.join(alifiles_dev), alifile_dev))

    if DNNFEATURESPRO:
        delta_order = config.get('nnet','delta_order')
        if int(delta_order) != 0:
            os.system(''' copy-feats scp:%s/feats_shuffled.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark:- |
                        add-deltas --delta-order=%d ark:- ark,scp:%s/feats_delta.ark,%s/feats_delat.scp''' %(
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'), int(delta_order),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name')))

            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark:- |
                        add-deltas --delta-order=%d ark:- ark,scp:%s/feats_delta.ark,%s/feats_delat.scp''' %(
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'), int(delta_order),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name')))

            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark:- |
                        add-deltas --delta-order=%d ark:- ark,scp:%s/feats_delta.ark,%s/feats_delat.scp''' %(
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), int(delta_order),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')))                                                                                    
            
        else:
            os.system(''' copy-feats scp:%s/feats_shuffled.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark,scp:%s/feats_delta.ark,%s/feats_delta.scp ''' %(
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name')))

            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark,scp:%s/feats_delta.ark,%s/feats_delta.scp ''' %(
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'dev_features') + '/' +  config.get('dnn-features', 'name')))

            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark,scp:%s/feats_delta.ark,%s/feats_delta.scp ''' %(
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')))

        #generate the global transform
        print '------- generate global transform ------'

        print "------- Using splice +/- $splice , step $splice_step ------"

        splice = int(config.get('nnet', 'splice'))
        splice_step = int(config.get('nnet', 'splice_step'))

        #save the current dir
        current_dir = os.getcwd()

        #go to kaldi egs dir
        os.chdir(config.get('directories', 'kaldi_egs'))

        os.system('''utils/nnet/gen_splice.py --fea-dim=%d --splice=%d > %s/tr_splice%d-%d.nnet ''' %(
                int(config.get('dnn-features','nfilt')), splice, 
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),
                splice, splice_step))
     
        #go back to working dir
        os.chdir(current_dir)

        os.system(''' head -n 3000 %s/feats_delta.scp > %s/train.scp.3k ''' %(
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name')))

        os.system(''' copy-feats scp:%s/train.scp.3k ark:- | add-deltas --delta-order=%s ark:- ark:- | 
                nnet-forward --use-gpu=yes %s/tr_splice%d-%d.nnet ark:- ark:- | compute-cmvn-stats ark:- - | cmvn-to-nnet - - |
                nnet-concat --binary=false %s/tr_splice%d-%d.nnet - %s/final.feature_transform ''' %(
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name'), config.get('nnet', 'delta_order'),
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'), splice, splice_step,
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'), splice, splice_step,
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),))

        os.system(''' copy-feats scp:%s/feats_delta.scp ark:- | nnet-forward --use-gpu=yes %s/final.feature_transform ark:- ark,scp:%s/final_train_feature.ark,%s/final_train_feature.scp''' %(
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'train_features') + '/' + config.get('dnn-features', 'name')))

        os.system(''' copy-feats scp:%s/feats_delta.scp ark:- | nnet-forward --use-gpu=yes %s/final.feature_transform ark:- ark,scp:%s/final_dev_feature.ark,%s/final_dev_feature.scp''' %(
                config.get('directories', 'dev_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),
                config.get('directories', 'dev_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'dev_features') + '/' + config.get('dnn-features', 'name')))

        os.system(''' copy-feats scp:%s/feats_delta.scp ark:- | nnet-forward --use-gpu=yes %s/final.feature_transform ark:- ark,scp:%s/final_test_feature.ark,%s/final_test_feature.scp''' %(
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name')))
    #create a target coder
    coder = target_coder.AlignmentCoder(lambda x, y: x, num_labels)

    #create a feature reader
    #create a train data feature reader
    featdir = config.get('directories', 'train_features') + '/' +  config.get('dnn-features', 'name')
    #create hdf5 formalt file
    if TRAINARKTOHDF5:
        ark2hdf5.HDF5(featdir+'/final_train_feature.scp', coder, alifile, config.get('directories','hdf5_train'),input_dim)

    with open(featdir + '/maxlength', 'r') as fid:
        max_input_length = int(fid.read())
        featreader = feature_reader.FeatureReader(featdir + '/final_train_feature.scp', max_input_length)
    #create a dev data feature reader
    featdir_dev = config.get('directories', 'dev_features') + '/' + config.get('dnn-features', 'name')
    if DEVARKTOHDF5:
        ark2hdf5.HDF5(featdir_dev+'/final_dev_feature.scp', coder, alifile_dev, config.get('directories','hdf5_dev'),input_dim)
    with open(featdir_dev + '/maxlength', 'r') as fid:
        max_input_length_dev = int(fid.read())
        featreader_dev = feature_reader.FeatureReader(featdir_dev + '/final_dev_feature.scp', max_input_length_dev)

    dispenser = batchdispenser.AlignmentBatchDispenser(featreader, coder, int(config.get('nnet', 'numutterances_per_block')), alifile)
    dispenser_dev = batchdispenser.AlignmentBatchDispenser(featreader_dev, coder, int(config.get('nnet', 'numutterances_per_block')), alifile_dev)
    #train the neural net
    print '------- training neural net ----------'
    nnet.train(config.get('directories','hdf5_train'), config.get('directories','hdf5_dev'), int(config.get('nnet','minibatch_size')), int(config.get('nnet','minibatch_size')))
    #compute the state prior and write it to the savedir
    prior = dispenser.compute_target_count().astype(np.float32)
    prior = prior + 1
    prior = prior/prior.sum()

    np.save(config.get('directories','expdir') + '/'+ config.get('nnet','name') + '/prior.npy', prior)


if TEST_NNET:

    delta_order = config.get('nnet','delta_order')
    if DNNTESTFEATPRO:
        if int(delta_order) != 0:
            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark:- |
                        add-deltas --delta-order=%d ark:- ark,scp:%s/feats_delta.ark,%s/feats_delat.scp''' %(
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), int(delta_order),
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                        config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')))  
        else:
            os.system(''' copy-feats scp:%s/feats.scp ark:- | apply-cmvn --print-args=false --norm-vars=%s --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp ark:- ark,scp:%s/feats_delta.ark,%s/feats_delta.scp ''' %(
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'), config.get('nnet', 'norm_vars'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name'),
                      config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')))

        os.system(''' copy-feats scp:%s/feats_delta.scp ark:- | nnet-forward --use-gpu=yes %s/final.feature_transform ark:- ark,scp:%s/final_test_feature.ark,%s/final_test_feature.scp''' %(
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'expdir') + '/' + config.get('nnet', 'name'),
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name'),
                config.get('directories', 'test_features') + '/' + config.get('dnn-features', 'name')))


    #use the neural net to calculate posteriors for the testing set
    print '------- computing state pseudo-likelihoods ----------'
    savedir = config.get('directories', 'expdir') + '/' + config.get('nnet', 'name')
    decodedir = savedir + '/decode'
    if not os.path.isdir(decodedir):
        os.mkdir(decodedir)

    featdir = config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name')

    #create a feature reader
    with open(featdir + '/maxlength', 'r') as fid:
        max_length = int(fid.read())
    featreader = feature_reader.FeatureReader(featdir + '/final_test_feature.scp', max_length)

    #create an ark writer for the likelihoods
    if os.path.isfile(decodedir + '/likelihoods.ark'):
        os.remove(decodedir + '/likelihoods.ark')
    writer = ark.ArkWriter(decodedir + '/feats.scp', decodedir + '/likelihoods.ark')

    #decode with te neural net
    nnet.decode(featreader, writer)

    print '------- decoding testing sets ----------'
    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/final.mdl', decodedir))
    os.system('cp -r %s %s' %(config.get('directories', 'expdir') + '/' + config.get('nnet', 'gmm_name') + '/graph', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/utt2spk', decodedir))
    os.system('cp %s %s' %(config.get('directories', 'test_features') + '/' +  config.get('dnn-features', 'name') + '/text', decodedir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'kaldi_egs'))

    #decode using kaldi
    os.system('%s/kaldi/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general', 'cmd'), config.get('general', 'num_jobs'), decodedir, decodedir, decodedir, decodedir))

    #get results
    os.system('grep WER %s/kaldi_decode/wer_* | utils/best_wer.sh' % decodedir)

    #go back to working dir
    os.chdir(current_dir)
