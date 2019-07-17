import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

valid_actions = [None, 'forward', 'left', 'right']
qvalues = {}
Alpha = 0.75
Gamma = 0.6
epsilon = 0.01




class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.successes = []
        global posreward 
        posreward = []
        global negreward 
        negreward = []
        global RewardRates
        RewardRates =[]
        
        
      
        
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.successes.append(0)
        global counter
      
        del posreward[:]
        del negreward[:]
        global posreward 
        posreward = []
        global negreward 
        negreward = []
       
       
        
    
        
     
        
        
   
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        try:
            counter
        except NameError:
            counter = 0
      
  
        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint) 
             
             
    
        #Consider this state the "s" state
        old_state = self.state
        print "OOOOLD", old_state
    
        # TODO: Select action according to your policy - this is the "a" action!
        best_oldaction = random.choice(valid_actions)
        try: qval = qvalues[(new_state, best_oldaction)]
        except : qval = 0
        for (state, action),values in qvalues.iteritems():
            if state == old_state and qvalues[(state, action)] > qval:
                best_oldaction = action #the new action a'
                qval = qvalues[(state, action)]
            else:
                 continue
        
        #epsilon greedy
        
        x=random.random()
        if(x<epsilon):
            reward = self.env.act(self, random.choice('left', 'right', 'forward')
        else:
            reward = self.env.act(self, best_oldaction)
            
        if reward >= 0:
            posreward.append(reward)
        else :
            negreward.append(reward)
        
       
        
        
        
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        new_state = (self.env.sense(self)['light'],self.env.sense(self)['oncoming'],self.env.sense(self)['left'],self.planner.next_waypoint()) # the state s'
        print "NEEEEEW", new_state
        
        # TODO: Learn policy based on state, action, reward
        
        
        best_newaction = random.choice(valid_actions)
        try: qval = qvalues[(new_state, best_newaction)]
        except : qval = 0
        for (state, action),values in qvalues.iteritems():
            if state == new_state and qvalues[(state, action)] > qval:
                best_newaction = action #the new action a'
                qval = qvalues[(state, action)]
            else:
                 continue
        
    
      
        
        if (new_state, best_newaction) not in qvalues:
            qvalues[(new_state, best_newaction)] = 0
        
        if (old_state, best_oldaction) in qvalues:
            qvalues[(old_state, best_oldaction)] = (1-Alpha)* qvalues[(old_state, best_oldaction)]+ Alpha*(reward+Gamma*qvalues[new_state,best_newaction])
        else:
            qvalues[(old_state, best_oldaction)] = 0
      
        
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]  
        if location == destination:
            self.successes[-1]=1
            print "SUCCESSES", self.successes
            print "Success Rate", float(sum(self.successes))/float(len(self.successes))
            print "Positive reward", posreward
            print "Negative reward", negreward
            print "Positive reward rate:", float(len(posreward))/(float(len(posreward))+float(len(negreward)))
            rate = float(len(posreward))/(float(len(posreward))+float(len(negreward)))
            RewardRates.append(rate)
            print "Pos Reward Rates!:", RewardRates
            print "Average Reward Rate over the last 10 trials:", float(sum(RewardRates[-10:]))/float(10.0)
            print "Q-TABLE:", qvalues
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, best_oldaction, reward)  # [debug]
    
        
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
