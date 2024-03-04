#include <HTTPClient.h>

// Placeholder for fetching sentiment data
void fetchSentimentData() {
  if(WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin("http://example.com/sentiment/BTC"); // Example sentiment API
    int httpCode = http.GET();
    
    if(httpCode > 0) {
        String sentimentPayload = http.getString();
        Serial.println(httpCode);
        Serial.println(sentimentPayload);
        // Parse the sentimentPayload JSON and extract the sentiment
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
    http.begin("http://api.coindesk.com/v1/bpi/currentprice/BTC.json"); 
    int httpCode = http.GET();
    
    if(httpCode > 0) {
        String pricePayload = http.getString();
        Serial.println(httpCode);
        Serial.println(pricePayload);
        // Parse the pricePayload JSON and extract the price
        // Consider calling fetchSentimentData() here or in the loop after fetching prices
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
  display.print("Crypto Price: "); // Placeholder text
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
  // Assuming fetchSentimentData() updates a global or passes data to updateDisplay
  fetchSentimentData();
  // Update the display with new data
  // You'll need to modify fetchCryptoPrices and fetchSentimentData to store their results in variables accessible here
  String price = "12345"; // Placeholder for actual price fetching
  String sentiment = "Positive"; // Placeholder for actual sentiment fetching
  updateDisplay(price, sentiment);
  delay(10000); // Fetch prices and sentiment every 10 seconds
}
