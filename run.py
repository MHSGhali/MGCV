import argparse
from two_dimensions.inf_focus import InfinityFocus


# def parse_args():
#     parent_parser = argparse.ArgumentParser('COCO Split/Visualize/Query')
#     subparsers = parent_parser.add_subparsers(title='actions', dest='command')
#     parser_split = subparsers.add_parser('split',add_help=True, description='The split parser', help='Split dataset to Training and Validation.')
#     parser_split.add_argument('-i', '--annotation_file', required=True, type=str, help='Path to the annotation file in COCO format.')
#     parser_split.add_argument('-s', '--split', default=0.8, type=float, help='Split ratio of training dataset.')
#     parser_split.add_argument('-o', '--output_path', required=True, type=str, help='Path to save training and validation jsons')

#     parser_viz = subparsers.add_parser('viz', add_help=True, description='The visualize parser', help='visualize coco dataset')
#     parser_viz.add_argument('-i', '--annotation_file', required=True, type=str, help='Path to the annotation file with COCO format.')
#     parser_viz.add_argument('-imp', '--image_paths', required=True, type=str, help='path to training image.')
#     parser_viz.add_argument('-o', '--output_path', type=str, help='Path to save overplotted masks, if not provided it it will just show  them using OpenCV!')
#     parser_viz.add_argument('-ss', '--sample_size', default=10, type=int, help='How many annotations to sample randomly for reviewing?')

#     parser_qry = subparsers.add_parser('query', add_help=True, description='The Query parser', help='Query images and annotations for coco dataset and specific categories. Your coco format needs to contain a url for the images!')
#     parser_qry.add_argument('-i', '--annotation_file', required=True, type=str, help='Path to the annotation file with COCO format.')
#     parser_qry.add_argument('-cat', '--categories', required=True, nargs='+', type=str, help='a list of categories to be queried. person dog etc.')
#     parser_qry.add_argument('-o', '--output_path', type=str, help='Path to save images and annotation json file!')

#     parser_convert = subparsers.add_parser('convert', add_help=True, description='The Converting parser', help='Convert all image annotations from COCO format to YOLO creating a text file for each image!')
#     parser_convert.add_argument('-i', '--annotation_file', required=True, type=str, help='Path to the annotation file with COCO format.')
#     parser_convert.add_argument('-o', '--output_directory', required=True, type=str, help='directory to output the annotation txt files')

#     args = parent_parser.parse_args()
#     return args


# def main():
#     args = parse_args()
#     if args.command == 'split':
#         cc = CustomCOCO(args.annotation_file, args.split, args.output_path)
#         cc.split_data()
#     elif args.command == 'viz':
#         scca = ShowCOCOAnnotations(annotation_file=args.annotation_file, images_path=args.image_paths, save_path=args.output_path, show_background=False, sample_size=args.sample_size)
#         scca.visualize_samples()
#     elif args.command == 'query':
#         qcoco = QueryCOCO(args.annotation_file)
#         qcoco.query_and_download_images(args.categories, args.output_path)
#     elif args.command == 'convert':
#         ccoco = Coco2YoloConverter(args.annotation_file)
#         ccoco.coco2yolo(output_path=args.output_directory)

image_path = "D:\\data_sets\\CV\\blending"
file_type = ".jpg"
    
InfinityFocus(image_path, file_type)