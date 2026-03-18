# L = 0.2126 * R + 0.7152 * G + 0.0722 * B

import torch

N = 1024


def main():
    torch.manual_seed(42)
    I = torch.randint(0, 255, (N, 3))
    # print(I, I.shape)
    O = torch.zeros(N)
    print(O, O.shape)
    for i, element in enumerate(I):
        L = 0.2126 * element[0] + 0.7152 * element[1] + 0.0722 * element[2]
        O[i] = L
    print(O)



if __name__ == "__main__":
    main()