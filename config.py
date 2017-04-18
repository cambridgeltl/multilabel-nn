class Defaults(object):
    home = "/home/sb/"
    encoding = 'utf-8'
    filter_sizes = [3,4,5]
    filter_nums = [200,200,200]
    sigmoid_threshold = 0.5
    hidden_sizes = []
    batch_size = 50
    drop_prob = 0.5    # None for no dropout
    l2_lambda = None    # None for no l2 regularization
    learning_rate = 0.001
    #token_regex = r'(\S+)'    # regular expression for whitespace tokenization
    token_regex = r'([^\W_]+|.)'    # fine-grained tokenization
    random_seed = 0xC001533D
    max_vocab_size = 2000000  # how many words to read from the embedding
    epochs = 100
    verbosity = 1
    optimizer = 'adam'
    fixed_embedding = False
    oversample = False    # balance class distribution by oversampling
    target_metric = 'fscore'
    threshold = False
    test = False
    oov = ''


    embedding_path = home+"embd/PubMed-and-PMC-w2v.bin"



    data_name = "hoc" #exp , hoc, mesh
    scope = "doc" #doc or sen (for document or sentence)

    expNames = ["ind","multi","init_UB"]
    expName= "init_UB"
    if data_name=="hoc": number_classes = 37
    if data_name == "exp": number_classes = 32
    if scope =="doc":doc_size = 500
    if scope =="sen":doc_size = 100

    input_dir = home+ "multilabel-nn/data/" # super dir where to find the input
    #results_dir = home+ "multilabel-nn/results/" # super dir where to ouput the results
    input_path = input_dir+scope+"/"+data_name+"/"
    output_path = home+ "multilabel-nn/out/" +scope+"/"+data_name+"/"+expName+"/"
    pred_path = home + "multilabel-nn/pred/" + scope + "/" + data_name + "/" + expName + "/"
    results_path = home+"multilabel-nn/res/"+scope+"/"+data_name+"/"+expName+"/"
    saved_mod_path = home + "multilabel-nn/saved_mod/" + scope + "/" + data_name + "/"+expName+"/"

    sigmoid_t_grid_start=0.05
    sigmoid_t_grid_stop = 0.95
    sigmoid_t_grid_step=0.05