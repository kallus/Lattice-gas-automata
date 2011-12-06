#include <stdlib.h>
#include <stdio.h>

int countBits(int magic) {
    magic = magic - ((magic >> 1) & 0x55555555);
    magic = (magic & 0x33333333) + ((magic >> 2) & 0x33333333);
    magic = (((magic + (magic >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

    return magic;
}

/* from http://snippets.dzone.com/posts/show/4716 */
void printbitssimple(int n) {
    unsigned int i;
    i = 1<<(1 * 8 - 1);

    while (i > 0) {
        if (n & i)
            printf("1");
        else
            printf("0");
        i >>= 1;
    }
}

int main(char argc, char **argv) {
    unsigned int i = 0;
    for (i = 0; i < 64; ++i) {
        printf("%i = ", i);
        printbitssimple(i);
        printf(" %i bits set.\n", countBits(i));
    }

    printf("static const unsigned char nSetBits[] = { ");
    for (i = 0; i < 64; ++i) {
        if (i % 8 == 0) {
            printf("\n  ");
        }
        printf("%i", countBits(i));
        if (i != 63) {
            printf(", ");
        }
    }
    printf("\n};\n");
    return 1;
}
