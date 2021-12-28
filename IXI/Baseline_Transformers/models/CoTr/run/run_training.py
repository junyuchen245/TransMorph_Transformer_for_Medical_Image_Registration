# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from CoTr.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from CoTr.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import os
import CoTr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=str, default='0')

    parser.add_argument("-network", type=str, default='3d_fullres')
    parser.add_argument("-network_trainer", type=str, default='nnUNetTrainerV2_ResTrans')
    parser.add_argument("-task", type=str, default='17', help="can be task name or task id")
    parser.add_argument("-fold", type=str, default='all', help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-outpath", type=str, default='Trainer_CoTr', help='output path')
    parser.add_argument("-norm_cfg", type=str, default='IN', help='BN, IN or GN')
    parser.add_argument("-activation_cfg", type=str, default='LeakyReLU', help='LeakyReLU or ReLU')

    parser.add_argument("-val", "--validation_only", default=False, help="use this if you want to only run the validation",
                        required=False, action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic", default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    norm_cfg = args.norm_cfg
    activation_cfg = args.activation_cfg
    outpath = args.outpath + '_' + norm_cfg + '_' + activation_cfg

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    if validation_only and (norm_cfg=='SyncBN'):
        norm_cfg=='BN'

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(outpath, network, task, network_trainer, plans_identifier, \
                                              search_in=(CoTr.__path__[0], "training", "network_training"), \
                                              base_module='CoTr.training.network_training')

    trainer = trainer_class(plans_file, fold, norm_cfg, activation_cfg, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    if args.disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_latest_checkpoint(train=False)

        trainer.network.eval()

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)

        if network == '3d_lowres':
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
