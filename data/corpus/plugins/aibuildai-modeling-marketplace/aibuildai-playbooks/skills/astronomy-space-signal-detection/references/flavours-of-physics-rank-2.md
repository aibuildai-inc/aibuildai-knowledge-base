# Second best solution

Competition: flavours-of-physics
Rank: #2
Source: https://www.kaggle.com/c/flavours-of-physics/discussion/17062

Source code and brief description of the solution ranked second on the Private Leaderboard can be found at https://github.com/gramolin/flavours-of-physics

I'd like to congratulate the Go Polar Bears team and Josef Slavicek who also achieved very high scores. I'm very grateful to the Organizers for the opportunity to work on an exciting problem and to analyze real data from LHCb. Here are some thoughts regarding the technical details of the competition which I'd like to share:

 1. I realize that it was difficult to perfectly combine real and simulated data and make them indistinguishable from one another. By removing real background events from the region of invariant masses around the tau mass you, of course, made sure that there are no potential decays of tau to three muons among them. At the same time, you introduced a serious vulnerability, because now mass can be used to perfectly discriminate between signal and background.
 2. It was probably a bad idea to provide the agreement and correlation datasets to participants. Instead, I think it would be enough to just let them check the KS and CvM scores while making submissions (limited number of times). This would ensure that they don't overuse the tests.
 3. For me personally, it was sad that you didn't provide azimuthal (phi) angles of the particles. This limited our ability to use kinematics to analyze events. At the same time, it didn't prevent us to calculate the mass of the mother particle anyway.

Despite all this, please don't stop your efforts to bring particle physicists and machine learning people together. We look forward to the next Kaggle competition from CERN!
