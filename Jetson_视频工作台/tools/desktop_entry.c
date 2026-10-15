/* Build on Jetson: gcc -O2 -Wall -Wextra tools/desktop_entry.c -o 双击启动网页 */
#include <limits.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef ENTRY_SCRIPT
#define ENTRY_SCRIPT "desktop_launcher.py"
#endif

int main(int argc, char **argv) {
    char path[PATH_MAX], script[PATH_MAX];
    ssize_t size = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (size < 0) return 1;
    path[size] = '\0';
    char *slash = strrchr(path, '/');
    if (!slash) return 1;
    *slash = '\0';
    if (snprintf(script, sizeof(script), "%s/tools/%s", path, ENTRY_SCRIPT) >= (int)sizeof(script)) return 1;
    char **args = calloc((size_t)argc + 3, sizeof(char *));
    if (!args) return 1;
    args[0] = "/usr/bin/python3";
    args[1] = "-I";
    args[2] = script;
    for (int i = 1; i < argc; ++i) args[i + 2] = argv[i];
    int headless = 0;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--no-ui") == 0) headless = 1;
    if (!headless) {
        /* The desktop entry may exit immediately; keep the GUI independent. */
        pid_t child = fork();
        if (child < 0) return 1;
        if (child > 0) return 0;
        if (setsid() < 0) return 1;
        int nullfd = open("/dev/null", O_RDWR);
        if (nullfd >= 0) {
            dup2(nullfd, STDIN_FILENO);
            dup2(nullfd, STDOUT_FILENO);
            dup2(nullfd, STDERR_FILENO);
            if (nullfd > STDERR_FILENO) close(nullfd);
        }
    }
    execv(args[0], args);
    execl("/usr/bin/zenity", "zenity", "--error", "--text=无法启动，请确认系统已安装 Python 3。", (char *)NULL);
    return 1;
}
