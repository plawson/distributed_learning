import glob
import sys
import os
import re


def split_sets(input_dir, file_pattern, number_for_train):

    if not os.path.exists(input_dir) and not os.path.isdir(input_dir):
        print("Cannot access feature files directory: {}!".format(input_dir))
        sys.exit(-1)

    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, 0o755, True)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, 0o755, True)
    """

    train_set = {}
    test_set = {}

    directory_list = glob.glob(input_dir + '/' + file_pattern)

    for featured_file in directory_list:
        file_name = os.path.basename(featured_file)
        key = file_name[0:re.search("\d", file_name).start() - 1]

        if train_set.get(key) is None:
            train_set[key] = []

        if len(train_set[key]) >= number_for_train:
            if test_set.get(key) is None:
                test_set[key] = []
            test_set[key].append(file_name)
        else:
            train_set[key].append(file_name)

    return train_set, test_set


def usage(prog):
    print("Usage: " + prog + " --dir <input_directory> --size <train_size_by_class>")


def main():
    i = 0
    args = sys.argv[1:]
    if len(args) != 4:
        usage(sys.argv[0])
        sys.exit(-1)

    input_dir = ""
    size = 0
    while i < len(args):
        if args[i] == "--dir":
            i += 1
            input_dir = args[i]
            if len(input_dir) == 0:
                print("Input directory name is required")
                sys.exit(-1)
        elif args[i] == "--size":
            i += 1
            if not args[i].isdigit():
                print("Train size by class ({}) must be a number".format(args[i]))
                sys.exit(-1)
            else:
                size = int(args[i])

        i += 1

    train_set, test_set = split_sets(input_dir, "*.json", size)

    print("train_set size: {}".format(len(train_set.keys())))
    print("train_set[0] size: {}".format(len(train_set[list(train_set)[0]])))

    print("test_set size: {}".format(len(test_set.keys())))
    print("test_set[0] size: {}".format(len(test_set[list(test_set)[0]])))


if __name__ == "__main__":
    main()
