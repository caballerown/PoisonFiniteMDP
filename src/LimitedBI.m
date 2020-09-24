function[z,U,A,w, Astar] = LimitedBI(R,P,B, T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon)

%cardS called k elsewhere in code for convenience
k=cardS;

%Find # of actions at each state
for s = 1:cardS
    A{s} = size(R{s},2);
end

%Initialize all reward to go to zero
U = zeros(T,cardS);

%Initialize terminal rewards
for s = 1:cardS
    U(T,s) = B{s}; 
end

%Limited BI Main Step -- Find action taken at tbar, sbar pair
for t=T-1:-1:tbar+1
    for s = 1:cardS
     for a =1:A{s}
        accumulator=0; %Used to iteratively cycle thru states & find summation
        for j=1:k-1
            accumulator(j) = P{s,a}(j)*(U(t+1,j)-U(t+1,k)); 
        end
        w{t,s}(a) = R{s}(a)+U(t+1,k)+sum(accumulator); %Candiate reward value for each state
     end
     %Find best rewarded action -- Use smooth max function (below) vice max()
     U(t,s) = max(w{t,s});
    end 
end

%Find Preferred Action at time t
t=tbar;
s=sbar;
 for a =1:A{sbar}
    accumulator=0; %Accumulates terms in summation
    for j=1:k-1
        accumulator(j) = P{s,a}(j)*(U(t+1,j)-U(t+1,k));
    end
    w{t,s}(a) = R{s}(a)+U(t+1,k)+sum(accumulator);
 end
 %Find preffered action-- used softmax vs. argmax
 [value_sbartbar, argmax] = max(w{t,s});
 
 Astar=0;
 if argmax == abar
     Astar = 1;
 end
 
%Find Objective Function
for i =1:size(Rhat,2)
    %Finds terms in the summation of each DV variable change
    diffVecb(i) = sum((Bhat{i}-B{i}).^2 / (Bhat{i}.^2 + epsilon) );
    diffvecR(i) = sum((Rhat{i} - R{i}).^2 / (Rhat{i}.^2 + epsilon) );
    for j =1:A{i}
        diffMatP(i,j) = sum((Phat{i,j}-P{i,j}).^2 / (Phat{i,j}.^2 + epsilon));
    end
end
%Calculates components of the obj function
d1 = sum(diffvecR);
d2 = sum(sum(diffMatP));
d3 = sum(diffVecb);
%Calculates actual obj func
z= -Astar +mu1*d1+mu2*d2+mu3*d3;

end
