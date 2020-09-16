#https://github.com/facebookresearch/clevr-aqa
import pickle, h5py, argparse, json, os, sys
sys.path.insert(0, os.path.abspath('.'))
print(os.path.abspath('.'))

import numpy as np
import aqa.programs
from aqa.preprocess import tokenize, encode, build_vocab


"""
Preprocessing script for ARGO question files.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='prefix', choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_questions_json', default='../output/ARGO_questions.json')
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=1, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)
parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='vocab.json')
parser.add_argument('--output_base',default='../output/processed')



def program_to_str(program, mode):
    if mode == 'chain':
        if not aqa.programs.is_chain(program):
            return None
        return aqa.programs.list_to_str(program)
    elif mode == 'prefix':
        program_prefix = aqa.programs.list_to_prefix(program)
        return aqa.programs.list_to_str(program_prefix)
    elif mode == 'postfix':
        program_postfix = aqa.programs.list_to_postfix(program)
        return aqa.programs.list_to_str(program_postfix)
    return None


def main(args):
    if not os.path.isdir(args.output_base):
        os.makedirs(args.output_base)
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        print('Must give one of --input_vocab_json or --output_vocab_json')
        return

    print('Loading data')
    with open(args.input_questions_json, 'r') as f:
        questions = json.load(f)['questions']

    all_program_strs = []
    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab == 1:
        print('Building vocab')
        ans=[]
        for q in questions:
            q['answer']=q['answer'].replace(' ','_')
        answer_token_to_idx = build_vocab(
            (q['answer'] for q in questions)
        )
        question_token_to_idx = build_vocab(
            (q['question'] for q in questions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )

        for q in questions:
            if 'program' not in q: continue
            program_str = program_to_str(q['program'], args.mode)
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = build_vocab(all_program_strs)
        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
        }

    if args.input_vocab_json != '':
        print('Loading vocab')
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab['question_token_to_idx']:
                if word not in vocab['question_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['question_token_to_idx'])
                    vocab['question_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        with open(os.path.join(args.output_base, args.output_vocab_json), 'w') as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    print('Encoding data')
    all_ques_strs = []
    questions_encoded, questions_length, programs_encoded = [], [], []
    question_family_idx = []
    template_file, orig_idxs = [], []
    image_idxs, video_names = [], []
    # question_class, question_family = [], []
    answers = []
    for orig_idx, q in enumerate(questions):
        question = q['question']
        all_ques_strs.append(question)
        orig_idxs.append(orig_idx)
        image_idxs.append(q['lidar_index'])
        video_names.append(q['video'])
        template_file.append(q['template_filename'])
        # question_class.append(q['question_class'])
        # question_family.append(q['question_family'])
        if 'question_family_index' in q:
            question_family_idx.append(q['question_family_index'])
        question_tokens = tokenize(question,
                                   punct_to_keep=[';', ','],
                                   punct_to_remove=['?', '.'])
        question_encoded = encode(question_tokens,
                                  vocab['question_token_to_idx'],
                                  allow_unk=args.encode_unk == 1)
        questions_encoded.append(question_encoded)
        questions_length.append(len(question_encoded))

        if 'program' in q:
            program = q['program']
            program_str = program_to_str(program, args.mode)
            program_tokens = tokenize(program_str)
            program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
            programs_encoded.append(program_encoded)

        if 'answer' in q:
            if type(q['answer']) == list:
                for ans in q['answer']:
                    answers.append(vocab['answer_token_to_idx'][ans.lower()])
            else:
                answers.append(vocab['answer_token_to_idx'][q['answer'].lower()])

    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])

    # Create h5 file
    print('Writing output')
    questions_encoded = np.asarray(questions_encoded, dtype=np.int64)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int64)
    questions_length = np.asarray(questions_length, dtype=np.int64)
    print(questions_encoded.shape)
    print(programs_encoded.shape)
    print(questions_length.shape)

    out_file = os.path.join(args.output_base, args.output_h5_file)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('video_names', data=np.string_(video_names))
        f.create_dataset('template_file', data=np.string_(template_file))
        f.create_dataset('questions', data=questions_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        f.create_dataset('question_length',data=questions_length)
        
        # if len(question_family) > 0:
        #     f.create_dataset('question_families', data=np.string_(question_family))
        # if len(question_class) > 0:
        #     f.create_dataset('question_classes', data=np.string_(question_class))
        if len(programs_encoded) > 0:
            f.create_dataset('programs', data=programs_encoded)
        if len(question_family_idx) > 0:
            f.create_dataset('question_family_idxs', data=np.asarray(question_family_idx))
        if len(answers) > 0:
            f.create_dataset('answers', data=np.asarray(answers))
    f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)