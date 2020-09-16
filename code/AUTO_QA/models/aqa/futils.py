import json, os, h5py, re
from gensim.models import Word2Vec
import gensim, importlib
import torch
import numpy as np
from gensim.models import KeyedVectors
import torch.nn.functional as F


def get_clevr_classes():
    classes = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '2', '3', '4', '5', '6', '7', '8', '9', 'animal', 'bicycle', 'bicyclist', 'bus', 'false', 'large_vehicle', 'moped', 'motorcycle', 'motorcyclist', 'obstacle', 'other_mover', 'pedestrian', 'trailer', 'true', 'vehicle']
    return classes

def get_one_hot_answers(answers, vocab):
    classes = get_clevr_classes()
    lookup = vocab['answer_idx_to_token']
    one_hot = np.zeros((answers.shape[0], len(classes)))
    for i, oh in enumerate(one_hot):
        c = lookup[answers[i]]
        c_idx = classes.index(c)
        oh[c_idx] = 1
    one_hot = one_hot.astype(np.int64)
    return one_hot

def get_answer_classes(answers, vocab):
    classes = get_clevr_classes()
    lookup = vocab['answer_idx_to_token']
    for i in range(answers.shape[0]):
        answers[i] = classes.index(lookup[answers[i]])
    answers = answers.astype(np.int64)
    return answers

def one_hot_to_answer(one_hot, vocab=None):
    classes = np.asarray(get_clevr_classes())
    c = classes[one_hot==1]
    assert c.shape[0] == 1
    c = c[0]
    if vocab is not None:
        vocab_idx = vocab['answer_token_to_idx'][c]
        return c,vocab_idx
    return c

def get_vocab(vocab_file):
    try:
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
    except:
        raise ValueError('File is not found in the location')
    return vocab


def get_word2vec(w2v_file):
    try:
        importlib.import_module('gensim', KeyedVectors)
    except:
        raise ModuleNotFoundError(' Gensim KeyedVectors module not found')
    else:
        try:
            w2v_vectors = KeyedVectors.load(w2v_file, mmap='r')
        except:
            raise ValueError('word2vec file not found')
    return w2v_vectors

def get_tokenize_expr():
    pat = r'[\w;]+'
    reg = re.compile(pat)
    return reg

def generate_w2v(context, size, window):
    try:
        importlib.import_module(gensim, Word2Vec)
    except:
        raise ModuleNotFoundError('Gensim Word2Vec module not found')
    
    vectors = Word2Vec(context, size=size, window=window, min_count=1, workers=4)
    return vectors

def generate_vectors(context, embedding_size, boundary_tokens=None, ):
    exp = get_tokenize_expr()
    all_context = []
    for c in context:
        cl = re.findall(exp, c)
        new_cl = [boundary_tokens[0]] if boundary_tokens is not None else []
        for ncl in cl:
            if ncl.endswith(';'):
                new_cl.append(ncl[:-1].lower())
                new_cl.append(';')
            else:
                new_cl.append(ncl.lower())
        if boundary_tokens is not None:
            new_cl.append(boundary_tokens[1])
        all_context.append(new_cl)
    w2v_model = generate_w2v(all_context, embedding_size, 7)
    return w2v_model


def correct_pred_count(pred, answer):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    correct_count_vector = (pred.data == answer.data)
    correct_count = correct_count_vector.sum()
    return correct_count_vector, correct_count


def test_base_masking(args, family_out):
    if not os.path.isfile(args.test_base):
        raise FileNotFoundError
    with open(args.test_base, 'r') as file:
        all_filters = json.load(file)
    output_file_name = args.model_type + '_' + args.features_type + '_' + args.encoder_type
    if args.mask_all:
        output_file_name = output_file_name + '_' + family_out
        if args.family_out not in all_filters.keys():
            raise ValueError("enter the valid family")
        all_filters = {family_out: all_filters[family_out]}
    return all_filters, output_file_name


def train_base_masking(args, family_out):
    if not os.path.isfile(args.train_base):
        raise FileNotFoundError
    with open(args.train_base, 'r') as file:
        all_filters = json.load(file)
    output_file_name = args.model_type + '_' + args.features_type + '_' + args.encoder_type
    output_file_name = output_file_name + '_' + family_out
    if family_out not in all_filters.keys():
        raise ValueError("enter the valid family")
    all_filters = all_filters.get(family_out)
    return all_filters, output_file_name

def save_model(args, epoch, loss, accuracy, model, optimizer, base_name='baseline'):
    name = base_name + '_Ep%d.pkl' % (epoch)
    print("Saving model " + name)
    
    if torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'model_state': model_state_dict,
        'optimizer_state': optimizer_state_dict
    }, os.path.join(args.model_dir, name))


def load_weights(args, model, optimizer=None):
    model_to_load = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(model_to_load):
        print('Model file not found at ', model_to_load)
        return model
    try:
        state_dict = torch.load(model_to_load, map_location=args.device)
        model_state_dict = state_dict['model_state']
        model.load_state_dict(model_state_dict)
        if optimizer is not None and 'optimizer_state' in model_state_dict.keys():
            optimizer_state_dict = state_dict['optimizer_state']
            optimizer.load_state_dict(optimizer_state_dict)
        epoch = state_dict['epoch']
        loss = state_dict['loss']
        accuracy = state_dict['accuracy']
        return [model, optimizer, epoch, loss, accuracy]
    except:
        print('Error occured while loading model')
        return model
    
    
    


def get_qa_from_idx(question, answer, vocab):
    classes = get_clevr_classes()
    ques = []
    for idx in question:
        if idx == 0:
            break
        ques.append(vocab['question_idx_to_token'][idx])
    ans = classes[answer]
    ques = ' '.join(ques[1:-1])
    return ques, ans


def get_vid_idx(video_dir):
    vid_idx = {}
    all_vids = os.listdir(video_dir)
    for name in sorted(all_vids):
        vid_idx[int(name.split('_')[-1])] = name
    return vid_idx

def invert_dict(d):
  return {v: k for k, v in d.items()}


def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
  # Sanity check: make sure <NULL>, <START>, and <END> are consistent
  assert vocab['question_token_to_idx']['<NULL>'] == 0
  assert vocab['question_token_to_idx']['<START>'] == 1
  assert vocab['question_token_to_idx']['<END>'] == 2
  assert vocab['program_token_to_idx']['<NULL>'] == 0
  assert vocab['program_token_to_idx']['<START>'] == 1
  assert vocab['program_token_to_idx']['<END>'] == 2
  return vocab
