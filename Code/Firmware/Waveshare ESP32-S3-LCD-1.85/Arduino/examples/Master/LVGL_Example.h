#pragma once

#include "LVGL_Driver.h"
#include "Display_ST77916.h"
#include "Gyro_QMI8658.h"
#include "RTC_PCF85063.h"
#include "SD_Card.h"
#include "Wireless.h"
#include "BAT_Driver.h"

#define EXAMPLE1_LVGL_TICK_PERIOD_MS  1000
extern uint8_t UI_Page;
extern lv_obj_t * t1;
extern lv_obj_t * t2;

void Page_switching(void);       

void Backlight_adjustment_event_cb(lv_event_t * e);

void Lvgl_Example1(void);
void LVGL_Backlight_adjustment(uint8_t Backlight);