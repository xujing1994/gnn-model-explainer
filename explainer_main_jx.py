""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain
import random
import utils.graph_utils as graph_utils
from torch.autograd import Variable
import numpy as np
import sys
import time



def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
    parser.add_argument(
        "--maaddir", dest="maaddir", help="Directory where marked adjency matrix is located"
    )
    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=True,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )
    parser.add_argument(
        "--poisoning-intensity",
        dest="poisoning_intensity",
        help="number of trigger samples in the training dataset"
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument('--device', dest='device', type=int, default=0, help='which device to use')

    # TODO: Check argument usage
    parser.set_defaults(
        datadir='data',
        logdir="log",
        ckptdir="ckpt",
        maaddir='maad',
        dataset="syn1",
        opt="adam",  
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
        poisoning_intensity=0.02,
        feature_type='default',
        num_workers=1,
        device=0,
        epochs=100,
    )
    return parser.parse_args()

def get_cg(args, graphs, max_nodes=0):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    for G in graphs:
        for u in G.nodes():
            G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    dataset_sampler = graph_utils.GraphSampler(
        graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Determine explainer mode
    graph_mode = (
        args.graph_mode
        or args.multigraph_class >= 0
        or args.graph_idx >= 0
    )
    ckpt = io_utils.load_ckpt(args)
    cg_dict_ori = ckpt["cg"] # get computation graph
    input_dim = cg_dict_ori["feat"].shape[2] 
    num_classes = cg_dict_ori["pred"].shape[2]

    if graph_mode: 
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    else:
        if args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).cuda() 
        # Explain Node prediction
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    if args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"]) 


    predictions = []
    for batch_idx, data in enumerate(dataset_loader):
        if batch_idx == 0:
            prev_adjs = data["adj"]
            prev_feats = data["feats"]
            prev_labels = data["label"]
            all_adjs = prev_adjs
            all_feats = prev_feats
            all_labels = prev_labels
        #elif batch_idx < 20:
        else:
            prev_adjs = data["adj"]
            prev_feats = data["feats"]
            prev_labels = data["label"]
            all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
            all_feats = torch.cat((all_feats, prev_feats), dim=0)
            all_labels = torch.cat((all_labels, prev_labels), dim=0)
        adj = Variable(data["adj"].float(), requires_grad=False).to(device)
        h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
        label = Variable(data["label"].long()).to(device)
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).to(device)

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        #if batch_idx < 5:
        #    predictions += ypred.cpu().detach().numpy().tolist()
        predictions += ypred.cpu().detach().numpy().tolist()

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset_loader))),
    }
    return cg_data

def sort_masked_adjs(masked_adjs):
    
    for i, maksed_adj in enumerate(masked_adjs):
        sorted_nodes = []
        node_list = list(range(masked_adjs[0].shape[0]))
        ind = np.dstack(np.unravel_index(np.argsort(masked_adjs[i], axis=None), masked_adjs[i].shape))
        for j in range(ind.shape[1]):
            if len(node_list) > 0:
                #print(masked_adjs[i][ind[0][ind.shape[1]-j-1][0]][ind[0][ind.shape[1]-j-1][1]])
                #print(ind[0][ind.shape[1]-j-1])
                if ind[0][ind.shape[1]-j-1][0] in node_list:
                    sorted_nodes.append(ind[0][ind.shape[1]-j-1][0])
                    node_list.remove(ind[0][ind.shape[1]-j-1][0])
                if ind[0][ind.shape[1]-j-1][1] in node_list:
                    sorted_nodes.append(ind[0][ind.shape[1]-j-1][1])
                    node_list.remove(ind[0][ind.shape[1]-j-1][1])
            else:
                break
        if i == 0:
            all_adjs = torch.as_tensor(sorted_nodes)
            all_adjs = all_adjs[np.newaxis, :]
        else:
            prev_adjs = torch.as_tensor(sorted_nodes)
            prev_adjs = prev_adjs[np.newaxis, :]
            all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)

    return all_adjs


def main():
    # Load a configuration
    prog_args = arg_parse()

    graphs = io_utils.read_graphfile(
        prog_args.datadir, prog_args.bmname
    )

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)

    # load original cg_data
    cg_dict_ori = ckpt["cg"] # get computation graph
    input_dim = cg_dict_ori["feat"].shape[2] 
    num_classes = cg_dict_ori["pred"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    # build model
    print("Method: ", prog_args.method)
    if graph_mode: 
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    else:
        if prog_args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).cuda() 
        # Explain Node prediction
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"]) 

    ## get compute graph of the dataset
    cg_dict = get_cg(prog_args, graphs, max_nodes=0)

    # Create explainer
    explainer = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=False,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
    )

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support 
    # We could even move each mode to a different method (even file)
    if prog_args.explain_node is not None:
        explainer.explain(prog_args.explain_node, unconstrained=False)
    elif graph_mode:
        if prog_args.multigraph_class >= 0:
            print(cg_dict["label"])
            # only run for graphs with label specified by multigraph_class
            labels = cg_dict["label"].numpy()
            graph_indices = []
            for i, l in enumerate(labels):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
                if len(graph_indices) > 30:
                    break
            print(
                "Graph indices for label ",
                prog_args.multigraph_class,
                " : ",
                graph_indices,
            )
            explainer.explain_graphs(graph_indices=graph_indices)

        elif prog_args.graph_idx == -1:
            # just run for a customized set of indices
            # masked_adjs = explainer.explain_graphs(graph_indices=[0, 1, 2, 3, 4])
            # trigger_idx = random.sample(range(cg_dict['label'].shape[0]), int(prog_args.poisoning_intensity*cg_dict['label'].shape[0]))
            #graphs_inds = range(cg_dict['label'].shape[0])
            graphs_inds = range(0, 3)
            masked_adjs = explainer.explain_all_graphs(graph_indices=graphs_inds)
            sorted_adjs = sort_masked_adjs(masked_adjs)
            os.makedirs(prog_args.maaddir, exist_ok=True)
            io_utils.save_masked_adjs(prog_args, sorted_masked_adjs=sorted_adjs)
        else:
            explainer.explain(
                node_idx=0,
                graph_idx=prog_args.graph_idx,
                graph_mode=True,
                unconstrained=False,
            )
            io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")
    else:
        if prog_args.multinode_class >= 0:
            print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            labels = cg_dict["label"][0]  # already numpy matrix

            node_indices = []
            for i, l in enumerate(labels):
                if len(node_indices) > 4:
                    break
                if l == prog_args.multinode_class:
                    node_indices.append(i)
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                node_indices,
            )
            explainer.explain_nodes(node_indices, prog_args)

        else:
            # explain a set of nodes
            masked_adj = explainer.explain_nodes_gnn_stats(
                range(400, 700, 5), prog_args
            )

def filter_adj(adjs):
    filt_adjs = adjs.copy()
    for filt_adj, adj in zip(filt_adjs, adjs):
        filt_adj[adj < 0.8] = 0
    return filt_adjs

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

