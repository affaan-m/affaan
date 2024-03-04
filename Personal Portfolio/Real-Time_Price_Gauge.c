#include <HTTPClient.h>

// fetch sentiment data we get from first running the python script to preprocess and use VADER then convert to JSON and process here via a HTTP Request
void fetchSentimentData() {
  if(WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin("http://example.com/sentiment/BTC"); //actual endpoint we will use is hosted elsewhere but not fully setup yet
    int httpCode = http.GET();
    
    if(httpCode > 0) {
        String sentimentPayload = http.getString();
        Serial.println(httpCode);
        Serial.println(sentimentPayload);
        // sentiment extraction step gather sentiment with endpoint in python first -> preprocessing done in python
    }
    else {
        Serial.println("Error on HTTP request for sentiment data");
    }
    http.end();
  }
}

void fetchCryptoPrices() {
  if(WiFi.status()== WL_CONNECTED){
    HTTPClient http;
    http.begin("http://api.coindesk.com/v1/bpi/currentprice/BTC.json"); // I might choose to fetch from CMC instead but API Costs are Prohibitive
    int httpCode = http.GET();
    
    if(httpCode > 0) {
        String pricePayload = http.getString();
        Serial.println(httpCode);
        Serial.println(pricePayload);
    }
    else {
        Serial.println("Error on HTTP request for price data");
    }
    http.end();
  }
}

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void setupDisplay() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);      
  display.setTextColor(SSD1306_WHITE);  
  display.setCursor(0,0);
  display.print("Crypto Price: ");
  display.display();
}

void updateDisplay(String price, String sentiment) {
  display.clearDisplay();
  display.setCursor(0,0);
  display.print("BTC Price: ");
  display.println(price);
  display.print("Sentiment: ");
  display.println(sentiment);
  display.display();
}

void setup() {
  Serial.begin(115200);
  // Initialize WiFi
  setupWiFi();
  // Initialize Display
  setupDisplay();
}

void loop() {
  fetchCryptoPrices();
  fetchSentimentData();
  String price = "12345";
  String sentiment = "Positive";
  updateDisplay(price, sentiment);
  delay(10000); //
}
