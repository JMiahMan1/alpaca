#include "lvgl.h"

#include <time.h>

void lvgl_ui_create(void);

int main(void)
{
    lv_init();

    lv_display_t *display = lv_sdl_window_create(480, 320);
    if (display == NULL) {
        return 1;
    }
    lv_sdl_mouse_create();
    lv_sdl_mousewheel_create();
    lv_sdl_keyboard_create();

    lvgl_ui_create();

    for (;;) {
        uint32_t idle = lv_timer_handler();
        if (idle > 10U) {
            idle = 10U;
        }
        struct timespec ts = {.tv_sec = 0, .tv_nsec = (long)idle * 1000000L};
        nanosleep(&ts, NULL);
    }
}
