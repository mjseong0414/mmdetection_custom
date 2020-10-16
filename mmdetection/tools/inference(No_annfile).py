from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import mmcv
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help= 'test config file path')
    parser.add_argument('--checkpoint', help= 'checkpoint file')

    args = parser.parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint

    model = init_detector(config_file, checkpoint_file, device= 'cuda:0')
    image_path = os.path.abspath("./data/person_test/")
    result_path = os.path.abspath("./data/image_inference_result01")
    images = sorted(os.listdir(image_path))
    integer = 0

    # images에 있는 image를 하나씩 뽑아가면서 inference 시켜야함
    for imgs in images:
        img_for_inference = image_path + "/" + imgs
        result = inference_detector(model, img_for_inference)
        last_path = result_path + "/{0:06d}".format(integer) + ".jpg"
        save_result_pyplot(model, img_for_inference, result, last_path)
        integer += 1

if __name__=='__main__':
    main()