"""ANSI color codes for the command line output."""

RESET = u"\u001b[0m"
BLACK = u"\u001b[30m"
RED = u"\u001b[31m"
GREEN = u"\u001b[32m"
YELLOW = u"\u001b[33m"
BLUE = u"\u001b[34m"
MAGENTA = u"\u001b[35m"
CYAN = u"\u001b[36m"
WHITE = u"\u001b[37m"
BRIGHT_BLACK = u"\u001b[30;1m"
BRIGHT_RED = u"\u001b[31;1m"
BRIGHT_GREEN = u"\u001b[32;1m"
BRIGHT_YELLOW = u"\u001b[33;1m"
BRIGHT_BLUE = u"\u001b[34;1m"
BRIGHT_MAGENTA = u"\u001b[35;1m"
BRIGHT_CYAN = u"\u001b[36;1m"
BRIGHT_WHITE = u"\u001b[37;1m"

BACKGROUND_BLACK = u"\u001b[40m"
BACKGROUND_RED = u"\u001b[41m"
BACKGROUND_GREEN = u"\u001b[42m"
BACKGROUND_YELLOW = u"\u001b[43m"
BACKGROUND_BLUE = u"\u001b[44m"
BACKGROUND_MAGENTA = u"\u001b[45m"
BACKGROUND_CYAN = u"\u001b[46m"
BACKGROUND_WHITE = u"\u001b[47m"
BACKGROUND_BRIGHT_BLACK = u"\u001b[40;1m"
BACKGROUND_BRIGHT_RED = u"\u001b[41;1m"
BACKGROUND_BRIGHT_GREEN = u"\u001b[42;1m"
BACKGROUND_BRIGHT_YELLOW = u"\u001b[43;1m"
BACKGROUND_BRIGHT_BLUE = u"\u001b[44;1m"
BACKGROUND_BRIGHT_MAGENTA = u"\u001b[45;1m"
BACKGROUND_BRIGHT_CYAN = u"\u001b[46;1m"
BACKGROUND_BRIGHT_WHITE = u"\u001b[47;1m"


def color_by_id(color_id):
    return u"\u001b[38;5;{}m".format(color_id)


def background_color_by_id(color_id):
    return u"\u001b[48;5;{}m".format(color_id)


if __name__ == '__main__':
    print(BRIGHT_BLACK + "hello world!" + RESET)
    print(BRIGHT_RED + "hello world!" + RESET)
    print(BRIGHT_GREEN + "hello world!" + RESET)

    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i*16+j)
            print(color_by_id(i*16+j) + " " + code.ljust(4), end="")
        print(RESET)

    for i in range(0, 16):
        for j in range(0, 16):
            code = str(i*16+j)
            print(background_color_by_id(i*16+j) + " " + code.ljust(4), end="")
        print(RESET)
