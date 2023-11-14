import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
import pickle
import time

from data.Config import *
from data.Dataloader import *
from modules.Parser import *
from modules.EDULSTM import *
from modules.Decoder import *
from modules.XLNetTune import *
from modules.TypeEmb import *
from data.TokenHelper import *
from modules.GlobalEncoder import *
from modules.Optimizer import *
from driver.TrainTest import predict, evaluate
from data.GumDataLoader import *

from torch.cuda.amp import autocast as autocast

def predictGUM(data, parser, vocab, config, token_helper, outputFile):
    start = time.time()
    parser.eval()
    with open(outputFile, mode='w', encoding='utf8') as outf:
        for onebatch in data_iter(data, config.test_batch_size, False):
            doc_inputs = batch_doc_variable(onebatch, vocab, config, token_helper)
            EDU_offset_index, batch_denominator, edu_lengths, edu_types = batch_doc2edu_variable(onebatch, vocab, config, token_helper)

            with autocast():
                parser.encode(
                    doc_inputs,
                    EDU_offset_index, batch_denominator, edu_lengths, edu_types
                )
                parser.decode(onebatch, None, None, vocab)

            for idx, inst in enumerate(onebatch):
                states = parser.batch_states[idx]
                step = parser.step[idx]
                pred_tree = getPredictRstTree(inst, vocab, states, step)
                outf.write(str(pred_tree._root_node))

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)
    print("torch version: ", torch.__version__)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='experiments/rst_model/config.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--test_dir', default='gum_rst_lisp_binary_c', help='without evaluation')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    discourse_parser_model = torch.load(config.load_model_path)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    print('Load pretrained encoder.....')
    token_helper = TokenHelper(config.save_plm_path)
    auto_extractor = AutoModelExtractor(config.save_plm_path, config, token_helper)
    print('Load pretrained encoder ok')

    global_encoder = GlobalEncoder(vocab, config, auto_extractor)
    EDULSTM = EDULSTM(vocab, config)
    typeEmb = TypeEmb(vocab, config)
    dec = Decoder(vocab, config)

    global_encoder.mlp_words.load_state_dict(discourse_parser_model["mlp_words"])
    global_encoder.rescale.load_state_dict(discourse_parser_model["rescale"])
    EDULSTM.load_state_dict(discourse_parser_model["EDULSTM"])
    typeEmb.load_state_dict(discourse_parser_model["typeEmb"])

    dec.load_state_dict(discourse_parser_model["dec"])

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        global_encoder = global_encoder.cuda()
        EDULSTM = EDULSTM.cuda()
        typeEmb = typeEmb.cuda()
        dec = dec.cuda()

    parser = DisParser(global_encoder, EDULSTM, typeEmb, dec, config)

    for file_name in os.listdir(args.test_dir):
        gum_data = []
        if file_name[-9:] != "crelation": continue
        gum_file_path = os.path.join(args.test_dir, file_name)
        print("file: ", gum_file_path)

        doc = read_gum_corpus_ll(gum_file_path)
        gum_data.append(doc)
        gum_inst = inst(gum_data)

        predictGUM(gum_inst, parser, vocab, config, token_helper, gum_file_path + '.out')
        #evaluate(gum_file_path, gum_file_path + '.out')

