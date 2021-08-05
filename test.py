from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.dataroot = "datasets/human_seg"
    opt.name = "human_seg"
    opt.arch ="meshunet"
    opt.dataset_mode ="segmentation"
    opt.ncf =[32, 64,128, 256] 
    opt.ninput_edges = 2280 
    opt.pool_res = [1800, 1350, 600]
    opt.resblocks = 3 
    opt.batch_size = 12 
    opt.export_folder = "meshes"
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
