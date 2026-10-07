#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

const char* WIFI_SSID = "ssid";
const char* WIFI_PASSWORD = "pass";

#define NUM_LED 6

int LED_PIN[NUM_LED] = {1,42,41,39,40,2};
int PWM_CHANNEL[NUM_LED] = {1,2,3,4,5,6};

#define PWM_FREQ 5000
#define PWM_RESOLUTION 8

WebServer server(80);

void setupPWM() {
  for(int i=0;i<NUM_LED;i++) {
    ledcSetup(PWM_CHANNEL[i],PWM_FREQ,PWM_RESOLUTION);
    ledcAttachPin(LED_PIN[i],PWM_CHANNEL[i]);
    ledcWrite(PWM_CHANNEL[i],0);
  }
}

void capture() {
  Serial.println("CAPTURE: requesting frame");
  camera_fb_t *fb=esp_camera_fb_get();

  if(!fb) {
    server.send(500,"text/plain","camera error");
    return;
  }

  Serial.print("CAPTURE: frame bytes=");
  Serial.println(fb->len);

  server.sendHeader("Access-Control-Allow-Origin","*");
  server.sendHeader("Connection","close");
  server.send_P(200,"image/jpeg",(char*)fb->buf,fb->len);

  esp_camera_fb_return(fb);
}

void lightControl() {
  if(!server.hasArg("id")||!server.hasArg("pwm")) {
    server.send(400,"text/plain","missing parameter");
    return;
  }

  int id=server.arg("id").toInt();
  int pwm=server.arg("pwm").toInt();

  if(id<0||id>=NUM_LED) {
    server.send(400,"text/plain","wrong id");
    return;
  }

  pwm=constrain(pwm,0,255);
  ledcWrite(PWM_CHANNEL[id],pwm);
  server.send(200,"text/plain","OK");
}

bool setupCamera() {
  camera_config_t config={};

  config.ledc_channel=LEDC_CHANNEL_0;
  config.ledc_timer=LEDC_TIMER_0;
  config.pin_d0=Y2_GPIO_NUM;
  config.pin_d1=Y3_GPIO_NUM;
  config.pin_d2=Y4_GPIO_NUM;
  config.pin_d3=Y5_GPIO_NUM;
  config.pin_d4=Y6_GPIO_NUM;
  config.pin_d5=Y7_GPIO_NUM;
  config.pin_d6=Y8_GPIO_NUM;
  config.pin_d7=Y9_GPIO_NUM;
  config.pin_xclk=XCLK_GPIO_NUM;
  config.pin_pclk=PCLK_GPIO_NUM;
  config.pin_vsync=VSYNC_GPIO_NUM;
  config.pin_href=HREF_GPIO_NUM;
  config.pin_sccb_sda=SIOD_GPIO_NUM;
  config.pin_sccb_scl=SIOC_GPIO_NUM;
  config.pin_pwdn=PWDN_GPIO_NUM;
  config.pin_reset=RESET_GPIO_NUM;
  config.xclk_freq_hz=20000000;
  config.pixel_format=PIXFORMAT_JPEG;
  config.frame_size=FRAMESIZE_VGA;
  config.jpeg_quality=10;
  config.grab_mode=CAMERA_GRAB_LATEST;
  config.fb_location=CAMERA_FB_IN_PSRAM;
  config.fb_count=2;

  esp_err_t err=esp_camera_init(&config);

  if(err!=ESP_OK) {
    Serial.print("Camera init error: ");
    Serial.println(err);
    return false;
  }

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("ESP32 SMART LIGHT CAMERA");

  if(!setupCamera()) {
    Serial.println("CAMERA FAILED");
    return;
  }

  Serial.println("Camera OK");
  setupPWM();

  WiFi.begin(WIFI_SSID,WIFI_PASSWORD);
  WiFi.setSleep(false);

  Serial.print("Connecting WiFi");

  while(WiFi.status()!=WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.on("/capture",capture);
  server.on("/light",lightControl);
  server.begin();

  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}

