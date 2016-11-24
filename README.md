# priorityindex

The priority index was developed by 510 an initiative by the Netherlands Red Cross
Code is available under the GPL license.

![alt tag](http://510.global/wp-content/uploads/2015/06/510-opengraph.png)


## Future steps

To improve the performance, and reduce the error, of the prediction model we will try the following:
-	[ ] Add new base line data especially on vulnerability and coping capacity (through a community level inform risk)
-	[ ] Add spatial correlation in the feature selection (now encoded in the x and y variable and distance from the first impact). Recompute the positions of the municipalities with respect to a system of reference that is parallel to the typhoon path and has the origin in the point of first impact
-	[ ] Add additional geographical features of higher resolution
-	[ ] Improve the windspeed computation
-	[ ] Work closer with in-country actors to get more complete building damage data and people affected data, and understand better the data collection methodology.
-	[ ] Training only on high damage areas
-	[x] Splitting partially and totally damaged
-	[ ] Extract features from the rainfall data
- [ ] Number of times a big amount of rain falls; or maximum rainfall in 30 minutes
-	[ ] Find windspeed data with a higher resolution and find new wind related features

To reduce the time needed to release a prediction on damage after a new typhoon:
-	[x] End-to-end scripting of all data collection, cleaning, aggregation and analysis steps
-	[ ] Reach agreements with data providers to get timely access to high resolution windspeed, rain, earthquake intensity and flood extend data.

To predict at a more detailed level the following is needed:
-	[ ] Discuss with in-country actors if and how a damage count at Barangay level (1 level down from municipality) can be done and made available.
-	[ ] Access to more Barangay level base line data.

To scale the methodology to other countries:
-	[ ] Search for reliable impact data for historical events (damage to houses, people affected).
-	[ ] If no reliable counts are available in the area, the use of other sources such as UNISAT damage classifications based on remote sensing, or proxy indicators for damage (i.e. through change in mobile phone usage, social media records of damage, crowdsourcing of humanitarian agencies in the field) can be evaluated.
