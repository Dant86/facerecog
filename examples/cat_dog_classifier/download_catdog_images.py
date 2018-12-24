from google_images_download import google_images_download
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DOG_URL = "https://www.google.com/search?tbm=isch&source=hp&biw=&bih=&ei=AFohXKQGxtaTBZuTtOAH&q=dog"
CAT_URL = "https://www.google.com/search?tbm=isch&sa=1&ei=BFohXICZJLWU1fAP7uqe2AQ&q=cat&oq=cat&gs_l=img.3..0i67l9j0.51164.51337..51492...0.0..0.90.170.2......1....1..gws-wiz-img.......35i39.ba9ofv6gxVQ"


DOG_SETTINGS = {"url": DOG_URL, "image_directory": "photos/dog", "print_urls": True}
CAT_SETTINGS = {"url": CAT_URL, "image_directory": "photos/cat", "print_urls": True}

res = google_images_download.googleimagesdownload()

# get dog photos
res.download(DOG_SETTINGS)

# get cat photos
res.download(CAT_SETTINGS)