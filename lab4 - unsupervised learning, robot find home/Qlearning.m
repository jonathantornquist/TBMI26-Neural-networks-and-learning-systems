%% Initialization
% Adds required folders to path
addpath('gwfunctions', 'helperfunctions');

%  Initialize the world, Q-table, and hyperparameters
world = 3;
gwinit(world);
s=gwstate;


Q=zeros(s.ysize,s.xsize,4);
%Q=ypos, xpos, action, som i instruktionen

%V_s=expected future optimal reward in all states, learns by time. 
V=zeros(s.ysize,s.xsize);

%1=down, 2=up, 3=right and 4=left
%End of the map transitions. 
Q(1,:,2)=-inf;
Q(s.ysize,:,1)=-inf;
Q(:,1,4)=-inf; %Can't go left in the first column in Q
Q(:,s.xsize,3)=-inf;



actions=[1,2,3,4];
prob_a = (1/4) * ones(4,1);
eps0=0.95;
gamma = 0.9;  %Initiate parameters 
n=0.2;
training = 300;
 

for i=1:training
gwinit(world);
s=gwstate;


%Pos = (y-coordinat, x-coordinat) 
old_state=s;


eps(i) = eps0 - (0.5*i/training); %Smaller and smaller epsilon for each interation
while old_state.isterminal==0  
    
[action, opt_action] = chooseaction(Q, old_state.pos(1), old_state.pos(2), actions, prob_a, eps(i)); %Given funktion
new_state=gwaction(action);
reward=new_state.feedback; %Receive feedback for action

%From lecture slides, updating Q. 
Q(old_state.pos(1),old_state.pos(2),action)=(1-n)*Q(old_state.pos(1),old_state.pos(2),action)+n*(reward+gamma*max(Q(new_state.pos(1),new_state.pos(2),:)));

old_state=new_state;

end


end 


[V, P]=max(Q,[],3); %Optimal V and policy. Uses the max-function on Q. 

gwdraw
gwdrawpolicy(P)


% surf(V)
% colorbar  %Our surfplot over V, see report
% title('V function values')


%% Training loop

%  Train the agent using the Q-learning algorithm.
%See code above. 

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

% gwinit(world);
% s=gwstate;
% y_pos=s.pos(1);
% x_pos=s.pos(2);
% gwdraw();
% while s.isterminal == 0
%   s = gwaction(P(y_pos,x_pos));
%   gwplotarrow([y_pos,x_pos], P(y_pos, x_pos)); %Draw walking way from optimal policy
%   y_pos=s.pos(1);
%   x_pos=s.pos(2);
% end
