from python_firebase_url_shortener.url_shortener import UrlShortener

class UrlShortener:
    def __init__(self):
        # Provide the authentication token and username for Firebase Google URL Shortener.
        # TODO: (Optionally) Feel free to create your own project and subdomain for your dynamic links.
        # TODO: 1. Create a new project and web service through Firebase Console (https://console.firebase.google.com/).
        # TODO: 2. Add dynamics link support to your new project (https://firebase.google.com/docs/dynamic-links/).
        # TODO: 3. The credentials below and URLs in the bash script should work anyway.
        auth_token = "AIzaSyD9gqDjU7szafFuNwD524ZBsVSPZ0tjD8Y"
        user_name = "geomos"

        # Create instance of URL Shortener.
        self.url_shortener = UrlShortener(auth_token, user_name)

    def shorten(self, initial_url):
        return self.url_shortener.get_short_link(initial_url)
