/*
  Sensor Glove Project - Converted to PYTHON CONTROL
 
  Receives gesture data (single chars) from a Python script
  via Serial communication.
 
  Controls:
  - 1x I2C OLED Display (SSD1306)
  - 4x LEDs
  - 1x Buzzer
 
  Expected Chars from Python:
  'A' -> Gesture A
  'B' -> Gesture B
  'C' -> Gesture C
  'D' -> Gesture D
  'E' -> Gesture E
  'F' -> Gesture F
  'W' -> Gesture "Write"  <-- NEW ADDITION
  'L' -> Gesture L
  'N' -> No gesture / Clear
*/

// --- Include Libraries ---
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- Setup the OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// --- Define Output Pins ---
#define LED_1_PIN 2
#define LED_2_PIN 3
#define LED_3_PIN 4
#define LED_4_PIN 5
#define BUZZER_PIN 6

// A variable to keep track of the last message to prevent flickering
String lastGesture = "";

void setup() {
  // Start serial for communication with Python
  Serial.begin(9600);

  // --- Initialize all OUTPUT pins ---
  pinMode(LED_1_PIN, OUTPUT);
  pinMode(LED_2_PIN, OUTPUT);
  pinMode(LED_3_PIN, OUTPUT);
  pinMode(LED_4_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // --- Initialize the OLED ---
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;); // Loop forever
  }

  // --- Show initial message on OLED ---
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("Glove"));
  display.println(F("Ready..."));
  display.display();
  
  delay(1000);
}

void loop() {
  String currentGesture = lastGesture;
  
  // Check if Python has sent any data
  if (Serial.available() > 0) {
    // Read the incoming character
    char receivedChar = Serial.read();

    // --- GESTURE RECOGNITION LOGIC (from Python) ---
    
    // Check for 'A'
    if (receivedChar == 'A') {
      currentGesture = "Hello";
      Serial.println("PYTHON SENT: A");
      
      setLEDs(1); 
      tone(BUZZER_PIN, 1000, 150); 
    }
    // Check for 'B'
    else if(receivedChar == 'B') {
      currentGesture = "How are you!";
      Serial.println("PYTHON SENT: B");
      
      setLEDs(2); 
      tone(BUZZER_PIN, 1000, 150); 
    }
    // Check for 'C'
    else if (receivedChar == 'C') {
      currentGesture = "Gesture C";
      Serial.println("PYTHON SENT: C");
      
      setLEDs(3); 
      tone(BUZZER_PIN, 1000, 150);
    }
    // Check for 'D'
    else if (receivedChar == 'D') {
      currentGesture = "Thank You";
      Serial.println("PYTHON SENT: D");
      
      setLEDs(4); 
      tone(BUZZER_PIN, 1000, 150);
    }
    // Check for 'E'
    else if (receivedChar == 'E') {
      currentGesture = "Sorry!!";
      Serial.println("PYTHON SENT: E");
      
      setLEDs(4); 
      tone(BUZZER_PIN, 1000, 150);
    }
    // Check for 'F'
    else if (receivedChar == 'F') {
      currentGesture = "Gesture F";
      Serial.println("PYTHON SENT: F");
      
      setLEDs(4); 
      tone(BUZZER_PIN, 1000, 150); 
    }
    // --- NEW: Check for 'W' (Write) ---
    else if (receivedChar == 'W') {
      currentGesture = "Write"; // Display text
      Serial.println("PYTHON SENT: W");
      
      setLEDs(4); // Turn on 4 LEDs (Modify this number if you want fewer LEDs)
      tone(BUZZER_PIN, 1200, 200); // Slightly higher/longer beep for distinction
    }
    // Check for 'L'
    else if (receivedChar == 'L') {
      currentGesture = "Gesture L";
      Serial.println("PYTHON SENT: L");
      
      setLEDs(2); 
      tone(BUZZER_PIN, 1000, 150);
    }
    // Check for 'N' (No gesture)
    else if (receivedChar == 'N') {
      currentGesture = "No Gesture Detect";
      Serial.println("PYTHON SENT: No Gesture");
      
      setLEDs(0); 
      noTone(BUZZER_PIN);
    }
    // --- Catch-all for unknown gestures ---
    else {
      // Keep previous gesture or do nothing to avoid screen flicker
      Serial.print("PYTHON SENT: Unknown char ");
      Serial.println(receivedChar);
    }
  }

  // 4. Update the OLED Display
  // Only update the screen if the message has changed
  if (currentGesture != lastGesture) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.print(currentGesture);
    display.display();
    
    lastGesture = currentGesture; // Remember what we just printed
  }
  
  delay(50); // Wait a short time
}

// --- HELPER FUNCTION ---
// Lights up 'count' number of LEDs
void setLEDs(int count) {
  digitalWrite(LED_1_PIN, (count > 0) ? HIGH : LOW);
  digitalWrite(LED_2_PIN, (count > 1) ? HIGH : LOW);
  digitalWrite(LED_3_PIN, (count > 2) ? HIGH : LOW);
  digitalWrite(LED_4_PIN, (count > 3) ? HIGH : LOW);
}