This code was original written by Tamara Berg, then extended by
James Hays (jhhays@cs.cmu.edu)


updated - 8/18/2008 - In previous archives, I had included 
flickrapi.py instead of flickrapi2.py which handles missing XML 
fields.

This code was original written by Tamara Berg, then extended by
James Hays (jhhays@cs.cmu.edu)

An older version of this code was used to generate the geotagged database for 
IM2GPS: estimating geographic information from a single image.
James Hays and Alexei A. Efros.  CVPR 2008.
http://graphics.cs.cmu.edu/projects/im2gps/

The code operates in two distinct stages:
  first querying for images, then downloading them.

--------------------
1) Querying
--------------------
The image query code is written in Python, using the Python Flickr
API interface which gives acknowledgements and credits in the top
of flickrapi2.py.  To use the Flickr API, you need an API key (see
http://www.flickr.com/services/api/) which you will need to enter
in get_imgs_geo_gps_search.py (line 40) in addition to changing 
the output path (line 71).

The query script, get_imgs_geo_gps_search.py, searches for Flickr
images with keywords listed in place_rec_queries.txt. Negative
constraints are found at the bottom of place_rec_queries.txt.

Querying FAQ

a) I don't care about geo-tagged images, I want them all.

As is, get_imgs_dyn_timeskip.py will only retrieve geotagged
images. To retrieve all images delete the
   has_geo = "1",
   accuracy="6",
constraints to the Flickr search API calls (lines 121 and 210).

b) Why does get_imgs_geo_gps_search.py look so complicated / run so
slow?

The Flickr API will disable your key if you query too rapidly, so
it makes sense to do large queries which return hundreds of
results.  But doing big queries is problematic, because there
seems to be a long existing and long known bug in the Flickr
search function- for any given search, after the 1500th or so
image, duplicates will start to appear.  You can get around this
by doing time bounded queries, but then you run into the problem
of having to do too many small queries.

Therefore get_imgs_geo_gps_search.py does queries within dynamically
sized time intervals, always trying to have about 400 results for
a query.  If few images are being found because you've done a rare
query, the time interval for queries will tend to expand.  If too
many images are being found the time interval will shrink.  As the
time window moves towards the present day it will tend to get
narrower, because the rate that people upload pictures to Flickr
seems to be increasing.

c) Will I get duplicate images?

Yes. If an image is tagged with two keywords that you are searching
for, that image will show up in both search results. Flickr images
have unique serial numbers that make it easy to identify duplicates.

d) Can I run the script in parallel to speed things up?

It's possible, but it's more likely to get your API key disabled.

--------------------
2) Downloading
--------------------
The image download code is written in Matlab.  It accesses images
on Flickr's http server instead of going through the API, and thus
doesn't require an API key.  It reads the text files produced by
get_imgs_geo_gps_search.py, downloads the photo, and saves all of
the image attributes (tags, interestingness, long/lat, etc...) as
a matlab cell string array in the comment field of each jpg.  Use
imfinfo() to read them later.

The main function is downloadphotos_int.m.  You'll need to set the
paths at the top. if you've changed any of the fields saved by
they query script you'll need to edit the download script since it
expects certain fields in certain order (see the source code for
an example).

Downloading FAQ

a) What size images will this get?

Currently the code will try and find the Flickr "Large" size
photo, which has max width or height of 1024.  Failing that it
will try to get the "Original" size photo.  If the "Original" is
larger than 1024 height/width it will be downsampled to 1024. If
it is smaller than 500 height or width it will be thrown away.
Otherwise the image will be kept.

A significant fraction of images are too small by this criteria
and thus are thrown away. An alternative strategy would be to
download only the default size images, which will always be 
available although somewhat small.

b) How are the images written to disk?

Since most file systems have trouble with thousands of files in a
directory, the images are put into a hierarchy of directories that
contain no more than 1000 images each.  The hierarchy is

base_db_path / keyword / numbered subdir / img_name

for example

Flickr_gps/Argentina/00015/315157387_c36ba74681_100_23812473@N00.jpg

The image filenames contain the photo id, secret, server id, and
owner which can be used to trace the .jpg back to its source on
Flickr.  See the source code for examples of how the URLs are
constructed.

c) What about all those annoying artsy borders that people put
around their Flickr photos?

The download script tries to remove them, see remove_frame.m

d) Can I run the download script in parallel?

Yes, I've run 15 copies in parallel in the past.  I wouldn't
recommend doing any more than this because Flickr could get mad at
us.  They're aware that researchers are using Flickr as a data
source but their main concern is that we don't impact the quality
of service for the millions of people who use Flickr.

To run multiple scripts in parallel you'll need to split up the
text files from the query process manually, then change the path
in downloadphotos_int.m for each call.

e) What are these libraries and mex files?

Matlab's image resizing is slow so I call the open_cv image resize
through a wrapper.  If that's not working you can just change the
function calls to fast_resize to Matlab's slower imresize().

f) What about copyrights?

It is worth noting that Flickr allows photographers to specify
Creative Commons licenses for their images instead of the default
"all rights reserved". This script saves the license info with
each .jpg file, so you can pick out Creative Commons images after
the fact (in my experience it's less than 10% of images) It is
also possible to restrict the search to images with certain
licenses at query time. See the Flickr API for details.