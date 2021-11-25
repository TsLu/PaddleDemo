#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2021-11-25 15:55
# @Author  : TsLu
# @usage   : mpirun -npernode 2 python random_dump.py

from __future__ import print_function
import os
import time
import logging
import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.WARNING)


class DemoRandomDump():
    """
    demo fot paddle random dump
    """

    def net(self, args=None):
        """
        create a simple net
        """
        self.train_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.all_vars = []
        user_var = []
        doc_var = []
        with fluid.program_guard(self.train_program, self.startup_program):
            with fluid.unique_name.guard():
                label = fluid.layers.data(
                    name="click",
                    shape=[-1, 1],
                    dtype="int64",
                    lod_level=1,
                    append_batch_size=False)
                for item in ['1001', '1002', '1003', '1004']:
                    item_var = fluid.layers.data(
                        name=item,
                        shape=[1],
                        dtype="int64",
                        lod_level=1)
                    user_var.append(item_var)
                    self.all_vars.append(item_var)
                for item in ['1005', '1006', '1007']:
                    item_var = fluid.layers.data(
                        name=item,
                        shape=[1],
                        dtype="int64",
                        lod_level=1)
                    doc_var.append(item_var)
                    self.all_vars.append(item_var)

                user_embs = []
                for var in user_var:
                    emb = fluid.layers.embedding(input=var,
                                                 size=[10, 14],
                                                 is_sparse=True,
                                                 is_distributed=True,
                                                 param_attr=fluid.ParamAttr(name="user_emb"))
                    bow = fluid.layers.sequence_pool(
                        input=emb,
                        pool_type='sum')
                    user_embs.append(bow)

                doc_embs = []
                for var in doc_var:
                    emb = fluid.layers.embedding(
                        input=var,
                        size=[10, 18],
                        is_sparse=True,
                        is_distributed=True,
                        param_attr=fluid.ParamAttr(name="doc_emb"))
                    bow = fluid.layers.sequence_pool(
                        input=emb,
                        pool_type='sum'
                    )
                    doc_embs.append(bow)

                user_concat = fluid.layers.concat(user_embs, axis=1)
                user_concat = fluid.layers.reshape(user_concat, [1, 48])
                doc_concat = fluid.layers.concat(doc_embs, axis=1)
                doc_concat = fluid.layers.reshape(doc_concat, [1, 48])

                user_fc = fluid.layers.fc(
                    input=user_concat,
                    size=48,
                    act='relu')
                doc_fc = fluid.layers.fc(
                    input=doc_concat,
                    size=48,
                    act='relu')
                self.similarity = fluid.layers.reduce_sum(
                    fluid.layers.elementwise_mul(user_fc, doc_fc),
                    dim=1,
                    keep_dim=True)

                prediction = fluid.layers.sigmoid(self.similarity)
                # cost auc_var
                cost = fluid.layers.log_loss(
                    input=prediction,
                    label=fluid.layers.cast(x=label, dtype='float32'))
                self.avg_cost = fluid.layers.mean(x=cost)

                auc_var, batch_auc_var, auc_states = fluid.layers.auc(
                    input=prediction,
                    label=label,
                    num_thresholds=2 ** 12,
                    slide_steps=20)

        return self.train_program, self.startup_program, self.avg_cost

    def train(self, args=None):
        """
        to train modle
        """
        train_program, startup_program, avg_cost = self.net()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fleet.init()
        scope = fluid.Scope()
        # optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(avg_cost, [scope])
        # params to dump
        dump_params = []
        for param in train_program.list_vars():
            if param.persistable:
                if "_generat" not in param.name:
                    dump_params.append(param.name)
                if "fc_" in param.name or "conv_" in param.name:
                    dump_params.append(param.name + "@GRAD")
            elif "RENAME" not in param.name:
                if "fc_" in param.name or "dropout_4.tmp_0" in param.name or "concat_" in param.name:
                    dump_params.append(param.name)
                if "sequence_pool_" in param.name and "tmp_1" not in param.name:
                    dump_params.append(param.name)
        start_day = time.strftime('%Y%m%d', time.localtime())
        output_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(output_path, "join_main_program.pbtxt"), "w") as fout:
            print(train_program, file=fout)
        cur_desc_file = 'cur_desc.prototxt'
        with open(os.path.join(output_path, cur_desc_file), "w") as fout:
            fout.write(str(train_program._fleet_opt["fleet_desc"]))

        train_program._fleet_opt["dump_fields"] = dump_params
        train_program._fleet_opt["dump_param"] = dump_params
        train_program._fleet_opt["dump_fields_path"] = output_path + \
            "/random_dump/join/" + start_day + "/" + "delta-1"

        train_info = []
        # init and start server
        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        # init and start worker
        elif fleet.is_worker():
            with fluid.scope_guard(scope):
                exe.run(startup_program)
            fleet.init_worker()
            train_data_path = output_path + '/train_data'
            train_data_files = []
            for filename in os.listdir(train_data_path):
                train_data_files.append(
                    os.path.join(train_data_path, filename))
            # fleet dataset
            label = fluid.layers.data(
                name="click",
                shape=[-1, 1],
                dtype="int64",
                lod_level=1,
                append_batch_size=False)
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
            dataset.set_use_var([label] + self.all_vars)
            dataset.set_parse_ins_id(True)
            dataset.set_batch_size(1)
            dataset.set_thread(3)
            dataset.set_filelist(train_data_files)
            # load train data
            dataset.load_into_memory()
            # dataset.local_shuffle()
            PASS_NUM = 1
            for pass_id in range(PASS_NUM):
                var_dict = {"loss": avg_cost}
                global var_dict

                class FetchVars(fluid.executor.FetchHandler):
                    def __init__(self, var_dict=None, period_secs=2):
                        super(FetchVars, self).__init__(
                            var_dict, period_secs=2)

                    def handler(self, res_dict):
                        train_info.extend(res_dict["loss"])
                exe.train_from_dataset(
                    program=train_program,
                    dataset=dataset,
                    debug=False,
                    scope=scope)
            dataset.release_memory()
            fleet.shrink_sparse_table()
            fleet.shrink_dense_table(0.01, 11)
            fleet.print_table_stat(0)
            fleet.clear_one_table(0)
            fleet.clear_model()
        fleet.stop_worker()
        return train_info


if __name__ == '__main__':
    obj = DemoRandomDump()
    obj.train()
