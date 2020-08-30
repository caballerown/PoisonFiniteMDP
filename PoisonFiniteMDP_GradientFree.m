function [z_final,time, R, P, B,U] = PoisonFiniteMDP_GradientFree(T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon, stepsizeR, stepsizeP, stepsizeB, n, zchangelimit)

%Description of Variables
%T = integer (# of time periods in finite-horizon MDP)
%tbar = integer (Between 1 & T - time for which action is being affected)
%sbar = integer (Between 1 & # states -- state action is being targeted)
%abar = integer (# of action at sbar, tbar desired to be taken)
%cardS = integer (# of states)
%alpha = term used in softmax and smoothmax approximations
%mu1 -- mu3: obj func weights for the respective changes to the parameters
%Rhat: baseline MDP immediate rewards
    %Rhat{s} = [rhat(s,action1),...,rhat(s,action_N)] 
    %        = [reward for action 1 @ s, ..., reward for action_N at s]
%Phat: baseline MDP transitions probs
    %Phat{s,a} = [phat(state 1|s,a), ...., phat(state card(S)-1|s,a)]
    %           = [transitioZn prob to state given s &a, ...] 
    %    *Note: Last state excluded to ensure sum of probabilities equal 1 later
%Bhat: baseline MDP terminal rewards
    %Bhat{s}= [b(1),...,b(card(S))] = [terminal reward state 1, ...]    
%epsilon: small positive # used in onj func to keep always defined
%stepsizeR: amount in random coord direction in R
%stepsizeP: "                                 " P
%stepsizeN: "                                 " B
%n = # of runs thru coord descent algorithm
%zchangelimit: delta in paper, small negative # near zero -- stops algrithm

tic  %start timingg

%Initiate Decision variables to original MDP values
R=Rhat;
P=Phat;
B=Bhat;

%Find number of actions available at each state
for s = 1:cardS
    A{s} = size(Rhat{s},2);
end

%Initialize reward-to-go values to zero
U = zeros(T,cardS);

%Number of last state in MDP
k=cardS;

%Initialize Algorithm Specific Parameters
iter=1; %iteration counter
z(iter)=0;
iter=iter+1;
zchange = 10000;  %Set change in obj func b/w iter high at first


%%%%%%%
% Find first z-values from baseline then start while loop
%z(iter) = LimitedBI(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
z(iter) = LimitedBISmooth(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);


% Begin while loop
for loop =1:n
startofnew(loop) = iter;
zchange =  10000;
while zchange>zchangelimit
    iter = iter+1;

%Initialize the Rplus, Rminus,..., Pminus
Rplus = R;
Rminus = R; 
Bplus = B;
Bminus=B;
Pplus=P;
Pminus=P;

%Randomly select a non-empty element for Rplus
randstate_rplus = randi([1,cardS]);  %Keep track of random element for later
buildRand_rplus= R{randstate_rplus}; %Select random state
randaction_rplus = randi([1,length(buildRand_rplus)]); %Keep track of rand action for later
Rand_rplus = buildRand_rplus(randaction_rplus); %Actual random value

%Randomly select a non-empty element for Rminus
randstate_rminus = randi([1,cardS]);  %Keep track of random element for later
buildRand_rminus= R{randstate_rminus}; %Select random state
randaction_rminus = randi([1,length(buildRand_rminus)]); %Keep track of rand action for later
Rand_rminus = buildRand_rminus(randaction_rminus); %Actual random value

%Randomly select element for Bplus and Bminus
randstate_bplus = randi([1,cardS]); %Keep track for later use
Rand_bplus = B{randstate_bplus}; 
randstate_bminus = randi([1,cardS]); %Keep track for later use
Rand_bminus = B{randstate_bminus}; 


%Randomly select a non-empty element for Pplus
randstate1_plus = randi([1,cardS]); %Randomly select state and record
buildrandP_action_plus = find(~cellfun('isempty', {P{randstate1_plus,:}})); %Find non-empty actions in P
randP_a_plus = datasample(buildrandP_action_plus,1); %Selected random action
randstate2_p_plus = randi([1,cardS-1]); %Randomly select state and record
Rand_pplus = P{randstate1_plus,randP_a_plus}(randstate2_p_plus);

%Randomly select a non-empty element for Pminus
randstate1_minus = randi([1,cardS]); %Randomly select state and record
buildrandP_action_minus = find(~cellfun('isempty', {P{randstate1_minus,:}})); %Find non-empty actions in P
randP_a_minus = datasample(buildrandP_action_minus,1); %Selected random action
randstate2_p_minus = randi([1,cardS-1]); %Randomly select state and record
Rand_pminus = P{randstate1_minus,randP_a_minus}(randstate2_p_minus);

%Build Coord direction components for Rplus, Bplus, Rminus and Bminus
Rplus{randstate_rplus}(randaction_rplus) = Rplus{randstate_rplus}(randaction_rplus) + stepsizeR;
Rminus{randstate_rminus}(randaction_rminus) = Rminus{randstate_rminus}(randaction_rminus) - stepsizeR;
Bplus{randstate_bplus} = Bplus{randstate_bplus} + stepsizeB;
Bminus{randstate_bminus} = Bminus{randstate_bminus} - stepsizeB;

%Build Coord directions Pplus; Make sure prob constraints not violated
if sum(P{randstate1_plus,randP_a_plus}) + stepsizeR <=1 && Pplus{randstate1_plus,randP_a_plus}(randstate2_p_plus) + stepsizeR <= 1
     Pplus{randstate1_plus,randP_a_plus}(randstate2_p_plus)=Pplus{randstate1_plus,randP_a_plus}(randstate2_p_plus) + stepsizeP;
end
%Build Coord directions Pminus; Make sure prob constraints not violated
if sum(P{randstate1_minus,randP_a_minus}) + stepsizeR <=1 && Pminus{randstate1_minus,randP_a_minus}(randstate2_p_minus) + stepsizeR <= 1
     Pminus{randstate1_minus,randP_a_minus}(randstate2_p_minus)=Pminus{randstate1_minus,randP_a_minus}(randstate2_p_minus) + stepsizeP;
end


%For all 6 combos of directions -- find obj func values (q7 is w/o chng)
%q1 = LimitedBI(Rplus,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q2 = LimitedBI(Rminus,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q3 = LimitedBI(R,Pplus,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q4 = LimitedBI(R,Pminus,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q5 = LimitedBI(R,P,Bplus,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q6 = LimitedBI(R,P,Bminus,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
%q7 = LimitedBI(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q1 = LimitedBISmooth(Rplus,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q2 = LimitedBISmooth(Rminus,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q3 = LimitedBISmooth(R,Pplus,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q4 = LimitedBISmooth(R,Pminus,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q5 = LimitedBISmooth(R,P,Bplus,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q6 = LimitedBISmooth(R,P,Bminus,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
q7 = LimitedBISmooth(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);


%List of each of the 7 choices above used
Rchoices = {Rplus,Rminus,R,R,R,R,R};
Pchoices = {P, P, Pplus, Pminus, P, P,P};
Bchoices = {B,B,B,B, Bplus,Bplus, B};

%ID the min value and its respective coord direction used 
[minvalue, minindex] = nanmin([q1,q2,q3,q4,q5,q6,q7]);

%Update list of iteration solutions
z(iter) = minvalue;   

%Check if relative change is satisied and leave loop if req'd
 zchange = ( z(iter-1)-z(iter) ) / abs(z(iter-1));
 if zchange<= zchangelimit
     break
 end 

%Assign new R,P, B values accordingly
R=Rchoices{minindex};
P=Pchoices{minindex};
B=Bchoices{minindex};


end 

end 


%If error found based on softmax, report last good found value
z= z(~isnan(z)); %%Error handling step as discussed in Sec 3.1 of text
z_final = min(z); %Used instead of error handling logic
time = toc; %End timing;
end