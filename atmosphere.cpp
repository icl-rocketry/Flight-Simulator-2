// definitions
#include <iostream>
#include <string>
#include <curl/curl.h> // make sure to install libcurl
#include "json.h" // and json library
using json = nlohmann::json;

// define standard atmosphere which can be used if there is no api connection
// ...

// fetch weather data
// TODO: change the URL and parameters to the openmeteo ones

// stores API data in a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    // iterates over the output and adds to a string
    ((std::string*)userp)->append((char*)contents, size * nmemb);
return size * nmemb;
}

// API setup
string getWeatherData(const string& city, const string& apiKey) {
CURL* curl;
CURLcode res;
string readBuffer;

// initialise the connecton
curl = curl_easy_init();
if (curl) {
    // define the URL
    string url = "http://api.openweathermap.org/data/2.5/weather?q=" + city + "&appid=" + apiKey + "&units=metric";

    // not sure what the rest of this does but we need it
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK)
    cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;

    curl_easy_cleanup(curl);
    }
    return readBuffer;
}

// parse the json output to a more usable structure
void parseWeatherData(const string& weatherData) {
    auto jsonData = json::parse(weatherData);

    string city = jsonData["name"];
    double temperature = jsonData["main"]["temp"];
    string weather = jsonData["weather"][0]["description"];

    cout << "City: " << city << endl;
    cout << "Temperature: " << temperature << "°C" << endl;
    cout << "Weather: " << weather << endl;
}

// main function (move this all to separate files when it's working)
int main() {
    string city;
    string apiKey = "INSERT_YOUR_API_KEY_HERE"; // Insert API key here

    cout << "Insert city name: ";
    getline(cin, city);

    string weatherData = getWeatherData(city, apiKey);

    if (!weatherData.empty()) {
        parseWeatherData(weatherData);
    } else {
        // if there's an issue with the connection
        cout << "Failed to get weather data." << endl;
        // TODO: automatically fall back on ISA model
    }

    return 0;
}

