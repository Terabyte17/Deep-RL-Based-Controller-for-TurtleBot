# Controller for Self Balancing and Locomotion of 2 wheeled robot
Branches-
<br>
main - for self balancing
<br>
locomotion - for moving around
<br>
pid - pid based controller
<br>
plane variation - under progress
<br>
### Environment Installation Instructions
make virtual env in your laptop (not in repository)
activate virtual env (do this otherwise tensorflow versions will mess up) :)
change directory to balancebot via terminal.
run command
~~~
pip install -e .
~~~
Then go back to main directory and run the dqn_baselines.py
also run these commands
~~~
pip install stable-baselines
pip install tensorflow==1.15
~~~

