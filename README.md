# Advanced-Lane-Finding
Condenced and edited version of https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding put all together into one file and prepared for actual use in a real robotics project. For one, all the differant modules have been edited and put together into one file, so that, when running on a bot, the code can run faster. This also makes editing SO much easier. I wrote the findroad() function which takes a screenshot of the screen, saves it, then opens it and finds the lane in the photo. The idea is that init() will open camdesk which will be connected to the camera on the gimbal. The incoming stream of video will be fullscreened and analyzed as fast as possible. One thing to note is that all this code will be run as a module by a main.py which will send results to the arduino and on to the acuators. Please submit any thoughts or edits: I need to finish this project within less than a year. 
