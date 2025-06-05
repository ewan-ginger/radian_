/*****************************************************************************
  | File        :   Simulated_Gesture.c
  
  | help        : 
    The provided LVGL library file must be installed first
******************************************************************************/
#include "Simulated_Gesture.h"


lv_obj_t *current_obj = NULL;
static lv_obj_t *NOW_Page;
static lv_style_t style_checked;

bool Boot_Flag = 0;              
bool PWR_Flag = 0;               
bool OK_Flag = 0;                
uint8_t Next_count = 0;           
uint8_t Switch_count = 0;     

bool is_scrollable_y(const struct _lv_obj_t * obj) {

  lv_dir_t dir = lv_obj_get_scroll_dir(obj);                          
  lv_scrollbar_mode_t mode = lv_obj_get_scrollbar_mode(obj);            
  bool can_scroll_y = (dir & LV_DIR_VER) != 0;                       
  bool scroll_allowed = (mode == LV_SCROLLBAR_MODE_ON) || (mode == LV_SCROLLBAR_MODE_AUTO);    
  if (!can_scroll_y || !scroll_allowed) {                              
    return false;
  }

  lv_obj_t * modifiable_obj = const_cast<lv_obj_t*>(obj);              
  lv_coord_t scroll_top = lv_obj_get_scroll_top(modifiable_obj);
  lv_coord_t scroll_bottom = lv_obj_get_scroll_bottom(modifiable_obj);
  return (scroll_top != 0 || scroll_bottom != 0);                    
}


void scroll_up_fixed(lv_obj_t * container, lv_coord_t pixels) {
  if(container){
    if(is_scrollable_y(container)){
      lv_obj_scroll_by(container, 0, -pixels, LV_ANIM_ON); 
    }

  }
}
void set_first_focus() {
  while(UI_Page != 1 && UI_Page != 2)                               
  {
    vTaskDelay(pdMS_TO_TICKS(100));
    printf("Please Wait : Now on page 0 or 3\r\n");
  }
  printf("First selection\r\n");
  uint8_t hierarchy = 0;                                            
  uint32_t Subcomponent_Num = 0;                                   
  lv_obj_t *first_child;
  if(UI_Page == 1)
    first_child = t1;
  else if(UI_Page == 2)
    first_child = t2;
  NOW_Page = first_child;
  Subcomponent_Num = lv_obj_get_child_cnt(first_child);            
  while(Subcomponent_Num)                                          
  {
    first_child = lv_obj_get_child(first_child, NULL);              
    Subcomponent_Num = lv_obj_get_child_cnt(first_child);           
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  if(first_child){
    if (current_obj) {
      lv_obj_remove_style(current_obj, &style_checked, 0);            
    }
    current_obj = first_child;                                        
    if(current_obj){
      lv_obj_add_style(current_obj, &style_checked, 0);                 
      lv_obj_scroll_to(NOW_Page, 0, 0, LV_ANIM_ON);                    
      lv_area_t coords;
      lv_obj_invalidate(NOW_Page);                                  
      lv_obj_get_coords(current_obj, &coords);
      // printf("X1: %d, Y1: %d, X2: %d, Y2: %d\n", coords.x1, coords.y1, coords.x2, coords.y2);
      if (coords.y2 > LCD_HEIGHT*7/10) {                          
        scroll_up_fixed(NOW_Page,coords.y2 - (LCD_HEIGHT*7/10));
      }
    }
    lv_obj_t *screen = lv_scr_act();
    lv_obj_invalidate(screen);
  }
}
void set_next_focus() {
  printf("Switch selected parts\r\n");
  lv_obj_t *next_child;
  if(current_obj)                                                     
  {
    uint32_t Subcomponent_Number = lv_obj_get_index(current_obj);       
    lv_obj_t *Parent = lv_obj_get_parent(current_obj);             
    if(Parent){                                                         
      uint32_t Subcomponent_Num = lv_obj_get_child_cnt(Parent);         
      if(Subcomponent_Number+1 >= Subcomponent_Num){                    
        while(Subcomponent_Number+1 >= Subcomponent_Num){              
          if(Parent == NOW_Page){                                       
            set_first_focus();                                        
            return;
          }  
          else{                                                        
            Subcomponent_Number = lv_obj_get_index(Parent);          
            Parent = lv_obj_get_parent(Parent);                         
            if (Parent == NULL) {                                       
              set_first_focus();
              return;
            }
            Subcomponent_Num = lv_obj_get_child_cnt(Parent);            
          }
          vTaskDelay(pdMS_TO_TICKS(10));
        }                                                               
        next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);   
        if(next_child)
        {
          Subcomponent_Num = lv_obj_get_child_cnt(next_child);           
          while(Subcomponent_Num){                                       
            next_child = lv_obj_get_child(next_child, NULL);             
            Subcomponent_Num = lv_obj_get_child_cnt(next_child);          
            vTaskDelay(pdMS_TO_TICKS(10));
          }
        }
      }
      else{                                                           
        if(Parent) 
        {
          if(Subcomponent_Number == 0)
            next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);
          else{
            Subcomponent_Num = lv_obj_get_child_cnt(Parent);
            if(Subcomponent_Number+1 <= Subcomponent_Num)
              next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);
            else
            {
              set_first_focus();                                                
              return;
            }
          }
        }
        else
        {
          set_first_focus();                                                
          return;
        }
      }
      if (next_child) {                                                   
        if (current_obj) {
          lv_obj_remove_style(current_obj, &style_checked, 0);           
        }
        current_obj = next_child;                                         
        if(current_obj){
          lv_obj_add_style(current_obj, &style_checked, 0);             
        }
        else
        {
          set_first_focus();                                                
          return;
        }
        lv_area_t coords;
        lv_obj_get_coords(current_obj, &coords);
        // printf("X1: %d, Y1: %d, X2: %d, Y2: %d\n", coords.x1, coords.y1, coords.x2, coords.y2);
        if (coords.y1 < LCD_HEIGHT/10) {                                 
          scroll_up_fixed(NOW_Page,coords.y1 - (LCD_HEIGHT/10));
        }
        else if (coords.y2 > LCD_HEIGHT*7/10) {                          
          scroll_up_fixed(NOW_Page,coords.y2 - (LCD_HEIGHT*7/10));
        }
        lv_obj_t *screen = lv_scr_act();
        lv_obj_invalidate(screen);
      } 
    }
    else                                                               
      set_first_focus();                                           
  }
  else                                                                  
    set_first_focus();                                                  
}


void set_next_focus_custom() {
  printf("Switch selected parts\r\n");
  static uint8_t Custom_Flag = 0;
  lv_obj_t *next_child;
  if(current_obj)                                                       
  {
    uint32_t Subcomponent_Number = lv_obj_get_index(current_obj);       
    lv_obj_t *Parent = lv_obj_get_parent(current_obj);                  
    if(Parent){                                                       
      uint32_t Subcomponent_Num = lv_obj_get_child_cnt(Parent);        
      if(Subcomponent_Number+1 >= Subcomponent_Num){                   
        Custom_Flag ++;
        if(Custom_Flag == 2){                                           
          Custom_Flag = 0;                                             
          set_first_focus();                                           
          return;
        } 
        while(Subcomponent_Number+1 >= Subcomponent_Num){               
          if(Parent == NOW_Page){                                      
            set_first_focus();                                          
            return;
          }  
          else{                                                        
            Subcomponent_Number = lv_obj_get_index(Parent);          
            Parent = lv_obj_get_parent(Parent);                         
            if (Parent == NULL) {                                    
              set_first_focus();
              return;
            }
            Subcomponent_Num = lv_obj_get_child_cnt(Parent);           
          }
          vTaskDelay(pdMS_TO_TICKS(10));
        }                                                               
        next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);  
        if(next_child)
        {
          Subcomponent_Num = lv_obj_get_child_cnt(next_child);            
          if(Subcomponent_Num){                                           
            next_child = lv_obj_get_child(next_child, NULL);              
            Subcomponent_Num = lv_obj_get_child_cnt(next_child);          
            vTaskDelay(pdMS_TO_TICKS(10));
          }
        }
      }
      else{                                                               
        if(Parent){
          if(Subcomponent_Number == 0)
            next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);
          else{
            Subcomponent_Num = lv_obj_get_child_cnt(Parent);
            if(Subcomponent_Number+1 < Subcomponent_Num)
              next_child = lv_obj_get_child(Parent, Subcomponent_Number+1);
            else{
              set_first_focus();                                                
              return;
            }
          }
        }
        else{
          set_first_focus();                                               
          return;
        }
      }
      if (next_child) {                                                   
        if (current_obj) {
          lv_obj_remove_style(current_obj, &style_checked, 0);           
        }
        current_obj = next_child;                                         
        if(current_obj){
          lv_obj_add_style(current_obj, &style_checked, 0);                
        }
        else
        {
          set_first_focus();                                                
          return;
        }
        lv_area_t coords; 
        if(current_obj){
          lv_obj_get_coords(current_obj, &coords);
          // printf("X1: %d, Y1: %d, X2: %d, Y2: %d\n", coords.x1, coords.y1, coords.x2, coords.y2);
          if (coords.y1 < LCD_HEIGHT/10) {                            
            scroll_up_fixed(NOW_Page,coords.y1 - (LCD_HEIGHT/10));
          }
          else if (coords.y2 > LCD_HEIGHT*7/10) {                          
            scroll_up_fixed(NOW_Page,coords.y2 - (LCD_HEIGHT*7/10));
          }
        }
        lv_obj_t *screen = lv_scr_act();
        lv_obj_invalidate(screen);
      } 
    }
    else                                                             
      set_first_focus();                                               
  }
  else                                                                 
    set_first_focus();                                                  
}



struct Simulated_XY touch_data = {0};

void Simulated_Touch(void) { 
  lv_area_t coords = {0}; 
  if(Boot_Flag == 0 && PWR_Flag == 0){                                      
    if(current_obj){
      if(OK_Flag){                                                            
        if(lv_obj_has_flag(current_obj, LV_OBJ_FLAG_CLICKABLE)){           
          lv_obj_get_coords(current_obj, &coords);
          if(coords.y2 != 0 || coords.x2 != 0){                              
            touch_data.x = coords.x1 + (coords.x2 - coords.x1)/2;
            touch_data.y = coords.y1 + (coords.y2 - coords.y1)/2;
            touch_data.points = 1;
          }
        }
        OK_Flag = 0;
      }
      else if(Switch_count || Next_count){
        if(Next_count) {    
          Next_count --;
          if(UI_Page == 2)
            set_next_focus_custom();
          else
            set_next_focus();
        }
        else if(Switch_count) {  
          Switch_count --;
          Page_switching();
          lv_obj_remove_style(current_obj, &style_checked, 0);                  
          current_obj = NULL;
        }
      }
    }
    else{                                                                       
      if(!OK_Flag){                                                             
        if(Next_count) {    
          Next_count --;
          set_first_focus();
        }
        if(Switch_count) {  
          Switch_count --;
          Page_switching();
        }
      }
      else{
        OK_Flag = 0;
      }
    }
  }
}

void TouchTask(void *parameter) {
  while(1){
    if(!digitalRead(PWR_KEY_Input_PIN)) {       
      PWR_Flag = 1;
      if(Boot_Flag)                       
        OK_Flag = 1;                     
    }  
    else{
      if(!OK_Flag && PWR_Flag == 1)           
        Switch_count ++;                 
      PWR_Flag = 0; 
    }
    if(!digitalRead(BOOT_KEY_PIN)){         
      Boot_Flag = 1;
      if(PWR_Flag)                          
        OK_Flag = 1;                        
    }
    else{                                   
      if(!OK_Flag && Boot_Flag == 1)        
        Next_count ++;                   
      Boot_Flag = 0;
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  vTaskDelete(NULL);
  
}
void Simulated_Touch_Init(){
  pinMode(BOOT_KEY_PIN, INPUT);    
  
  lv_style_init(&style_checked);
  lv_style_set_bg_color(&style_checked, lv_color_white());
  lv_style_set_bg_opa(&style_checked, LV_OPA_COVER);
  lv_style_set_border_color(&style_checked, lv_palette_main(LV_PALETTE_GREY));
  lv_style_set_border_width(&style_checked, 1);
  lv_style_set_border_opa(&style_checked, LV_OPA_COVER);
  lv_style_set_pad_all(&style_checked, 6);
  lv_style_set_radius(&style_checked, 8);
  lv_style_set_shadow_color(&style_checked, lv_color_black());
  lv_style_set_shadow_width(&style_checked, 8);
  lv_style_set_shadow_spread(&style_checked, 0);
  lv_style_set_shadow_ofs_x(&style_checked, 4);
  lv_style_set_shadow_ofs_y(&style_checked, 4);

  xTaskCreatePinnedToCore(
    TouchTask,    
    "TouchTask",   
    4096,                
    NULL,                
    4,                    
    NULL,                
    1                  
  );
}





