import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--enable_unk_label_obj', action='store_true')
    p.add_argument('--use_valid_mask', action='store_true')
    p.add_argument('--use_feature_align', action='store_true')
    p.add_argument('--use_vlm_distill', action='store_true')
    p.add_argument('--clip_text_features', default=None)
    p.add_argument('--enable_unk_head', action='store_true')
    p.add_argument('--train_unk_head', action='store_true')
    p.add_argument('--infer_with_unk_head', action='store_true')
    p.add_argument('--unk_loss_use_known_neg', action='store_true')
    p.add_argument('--unk_loss_use_dummy_neg', action='store_true')
    p.add_argument('--unk_loss_use_dummy_pos', action='store_true')
    p.add_argument('--enable_unknown_output', dest='enable_unknown_output', action='store_true')
    p.add_argument('--disable_unknown_output', dest='enable_unknown_output', action='store_false')
    p.set_defaults(enable_unknown_output=True)
    return p


def main(argv=None):
    args, _ = build_parser().parse_known_args(argv)
    issues = []
    warns = []

    if args.use_vlm_distill and not args.enable_unk_label_obj:
        warns.append('use_vlm_distill is enabled without enable_unk_label_obj: VLM weighting only helps when dummy positives are being mined.')

    if (args.use_feature_align or args.use_vlm_distill) and not args.clip_text_features:
        warns.append('CLIP-related options are enabled but clip_text_features is not provided.')

    if args.train_unk_head and not args.enable_unk_head:
        issues.append('train_unk_head requires enable_unk_head.')

    if args.infer_with_unk_head and not args.enable_unk_head:
        issues.append('infer_with_unk_head requires enable_unk_head.')

    if args.enable_unk_head and args.train_unk_head and not args.enable_unk_label_obj:
        warns.append('unknownness head is trained without enable_unk_label_obj: there may be no reliable positive unknown samples.')

    if args.enable_unk_head and args.train_unk_head:
        if not (args.unk_loss_use_dummy_pos or args.unk_loss_use_known_neg or args.unk_loss_use_dummy_neg):
            issues.append('unknownness head training has no selected supervision source; enable at least one of dummy_pos / known_neg / dummy_neg.')
        if not args.unk_loss_use_dummy_pos:
            warns.append('unknownness head is trained without dummy positive supervision; the head may have no positive unknown targets.')

    if issues:
        print('INVALID CONFIG:')
        for x in issues:
            print(f'  - {x}')
        if warns:
            print('\nWARNINGS:')
            for x in warns:
                print(f'  - {x}')
        sys.exit(2)

    print('CONFIG OK')
    if warns:
        print('WARNINGS:')
        for x in warns:
            print(f'  - {x}')


if __name__ == '__main__':
    main()
