from utils import *
import argparse
import shutil
import os


def main(s3_bucket, s3_train_data, s3_logs_path, s3_tensorboard_path, s3_model_path, s3_mlflow_path, interm_sup,
         valid_out_dir,
         restore_model, start_ite, gt_encoder_model, model_digit, seed, cache_size, cache_boost_train,
         cache_boost_valid, input_size, crop_size, random_flip_h, random_flip_v, early_stop, model_save_fre,
         batch_size_train, batch_size_valid, max_ite, max_epoch_num):
    print(s3_bucket, s3_train_data, s3_logs_path, s3_tensorboard_path, s3_model_path, interm_sup, valid_out_dir,
          restore_model, start_ite, gt_encoder_model, model_digit, seed, cache_size, cache_boost_train,
          cache_boost_valid, input_size, crop_size, random_flip_h, random_flip_v, early_stop, model_save_fre,
          batch_size_train, batch_size_valid, max_ite, max_epoch_num)
    # --- Step 0: Build datasets ---
    # download data from s3
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        download_folder_from_s3(s3_bucket, s3_train_data, "./data")
    # make sure cache folder exists
    Path(os.path.sep.join([os.getcwd(), "data/train/cache"])).mkdir(parents=True,
                                                                    exist_ok=True)
    Path(os.path.sep.join([os.getcwd(), "data/val/cache"])).mkdir(parents=True,
                                                                  exist_ok=True)
    # make sure mlflow is updated
    download_mlruns(s3_bucket, s3_mlflow_path, os.getcwd())

    # get train and val directories
    train_dir = os.path.join(os.path.join(os.getcwd(), "data/train"))
    val_dir = os.path.join(os.path.join(os.getcwd(), "data/val"))
    train_datasets, valid_datasets = create_inputs(train_dir, val_dir)
    train_datasets, valid_datasets = create_inputs(train_dir, val_dir)

    # --- Step 1: Build dataloaders ---
    print("--- create training dataloader ---")
    # collect training dataset
    train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    # build dataloader for training datasets
    train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                           cache_size=cache_size,
                                                           cache_boost=cache_boost_train,
                                                           my_transforms=[
                                                               GOSRandomHFlip(),
                                                               GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                                                           ],
                                                           batch_size=batch_size_train,
                                                           shuffle=True)
    train_dataloaders_val, train_datasets_val = create_dataloaders(train_nm_im_gt_list,
                                                                   cache_size=cache_size,
                                                                   cache_boost=cache_boost_train,
                                                                   my_transforms=[
                                                                       GOSNormalize([0.5, 0.5, 0.5],
                                                                                    [1.0, 1.0, 1.0]),
                                                                   ],
                                                                   batch_size=batch_size_valid,
                                                                   shuffle=False)
    print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    # build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    # build dataloader for training datasets
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                           cache_size=cache_size,
                                                           cache_boost=cache_boost_valid,
                                                           my_transforms=[
                                                               GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                                                               # GOSResize(input_size)
                                                           ],
                                                           batch_size=batch_size_valid,
                                                           shuffle=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net = ISNetDIS()

    # convert to half precision
    if (model_digit == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if (restore_model != ""):
        print("restore model from:")
        model_path = os.path.join(os.getcwd(), 'weights')
        print(model_path + "/" + restore_model)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path + "/" + restore_model))
        else:
            net.load_state_dict(torch.load(model_path + "/" + restore_model, map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # --- Step 4: applying Distributed Data Parallelism ---
    net = nn.DataParallel(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # --- Step 3: Setup Mlflow tracking ---
    # mlflow_path = os.path.join(os.getcwd(), 'mlruns')
    mlflow.set_tracking_uri("file:/home/aos-mo/Documents/ML_models/dis_2/mlruns")
    mlflow.set_experiment("DIS")

    # --- Step 4: Train and Valid Model with MLFlow logs ---

    # Start a run with the next available run name
    current_run_number = mlflow.get_experiment_by_name("DIS").tags.get("current_run_number", "0")
    current_run_number = int(current_run_number)

    # Determine the next available run name
    next_run_name = f"dis-ff-{current_run_number}"
    mlflow.set_experiment("DIS")

    with mlflow.start_run(run_name=next_run_name) as run:
        # Set the run name tag
        mlflow.set_tag("mlflow.runName", next_run_name)
        # Log hyperparameters
        mlflow.log_params({
            "max_epochs": max_epoch_num,
            "train_batch_size": batch_size_train,
            "val_batch_size": batch_size_valid,
            "max_iterations": max_ite,
            "early_stop": early_stop,
            "model_save_frequency": model_save_fre
        })
        # Increment the run number for the next run
        current_run_number += 1
        mlflow.set_experiment("DIS")
        mlflow.set_experiment_tags({"current_run_number": str(current_run_number)})

        train(net,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              train_dataloaders_val, train_datasets_val,
              interm_sup, start_ite, gt_encoder_model, model_digit, seed, early_stop, model_save_fre, batch_size_train,
              batch_size_valid, max_ite, max_epoch_num, valid_out_dir, s3_bucket)

        # generate predictions
        model_path = os.path.join(os.getcwd(), 'weights')
        model_list = os.listdir(model_path)
        model_list.sort(key=lambda x: int(x.split('_') [2]))
        restore_model = model_list [-1]
        valid_out_dir = os.getcwd()

        # rebuild model for validation
        print("--- rebuild model for validation ---")
        # net = ISNetDIS()
        #
        # # convert to half precision
        # if (model_digit == "half"):
        #     net.half()
        #     for layer in net.modules():
        #         if isinstance(layer, nn.BatchNorm2d):
        #             layer.float()
        #
        # if torch.cuda.is_available():
        #     net.cuda()

        if (restore_model != ""):
            print("restore model from:")
            print(model_path + "/" + restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(model_path + "/" + restore_model))
            else:
                net.load_state_dict(
                    torch.load(model_path + "/" + restore_model, map_location="cpu"))

        valid(net, valid_dataloaders, valid_datasets, model_digit, batch_size_valid, max_epoch_num,
              valid_out_dir, epoch=0)

        pred_path = os.path.join(os.getcwd(), 'VAL-DATA')
        gt_path = os.path.join(os.getcwd(), 'data/val/gt')

        hce = compute_hce(pred_path, gt_path, "")
        mlflow.log_metric("hce", hce)

    # --- Step 5: upload to S3 and delete after---
    upload_folder_to_s3(os.path.sep.join([os.getcwd(), "logs"]), s3_bucket, s3_logs_path)
    shutil.rmtree(os.path.join(os.getcwd(), 'logs'))
    upload_folder_to_s3(os.path.sep.join([os.getcwd(), "weights"]), s3_bucket, s3_model_path)
    shutil.rmtree(os.path.join(os.getcwd(), 'weights'))
    upload_folder_to_s3(os.path.sep.join([os.getcwd(), "tensorboard"]), s3_bucket,
                        s3_tensorboard_path)
    shutil.rmtree(os.path.join(os.getcwd(), 'tensorboard'))
    upload_folder_to_s3(os.path.sep.join([os.getcwd(), "mlruns"]), s3_bucket,
                        s3_mlflow_path)
    shutil.rmtree(os.path.join(os.getcwd(), 'mlruns'))
    # shutil.rmtree(os.path.join(os.getcwd(), 'VAL-DATA'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_bucket', type=str, default='dis-sagemaker')
    parser.add_argument('--s3_train_data', type=str, default='fast-flow/data_test')
    parser.add_argument('--s3_logs_path', type=str, default="fast-flow/experiment-ec2/outputs/logs/train_logs")
    parser.add_argument('--s3_tensorboard_path', type=str, default="fast-flow/experiment-ec2/outputs/tensorboard")
    parser.add_argument('--s3_model_path', type=str, default="fast-flow/experiment-ec2/outputs/saved_weights")
    parser.add_argument('--s3_mlflow_path', type=str, default="mlruns")
    parser.add_argument('--interm_sup', default=False)
    parser.add_argument('--valid_out_dir', default="")
    parser.add_argument('--restore_model', default="")
    parser.add_argument('--start_ite', default=0)
    parser.add_argument('--gt_encoder_model', default="")
    parser.add_argument('--model_digit', default="full")
    parser.add_argument('--seed', default=0)
    parser.add_argument('--cache_size', default=[1024, 1024])
    parser.add_argument('--cache_boost_train', default=False)
    parser.add_argument('--cache_boost_valid', default=False)
    parser.add_argument('--input_size', default=[1024, 1024])
    parser.add_argument('--crop_size', default=[1024, 1024])
    parser.add_argument('--random_flip_h', default=1)
    parser.add_argument('--random_flip_v', default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--model_save_fre', type=int, default=5)
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=1000)
    parser.add_argument('--max_epoch_num', type=int, default=10)

    args = parser.parse_args()
    main(args.s3_bucket, args.s3_train_data, args.s3_logs_path, args.s3_tensorboard_path, args.s3_model_path,
         args.s3_mlflow_path, args.interm_sup, args.valid_out_dir, args.restore_model, args.start_ite,
         args.gt_encoder_model, args.model_digit, args.seed, args.cache_size, args.cache_boost_train,
         args.cache_boost_valid, args.input_size, args.crop_size, args.random_flip_h, args.random_flip_v,
         args.early_stop, args.model_save_fre, args.batch_size_train, args.batch_size_valid, args.max_ite,
         args.max_epoch_num)
