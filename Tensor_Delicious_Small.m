clear all;
Data=load('~/Documents/Data/MultiLabel/Delicious/Delicious_train_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;
Label=load('~/Documents/Data/MultiLabel/Delicious/Delicious_train_label.txt');
Label(:,1)=Label(:,1) + 1; Label(:,2)=Label(:,2)+1;

nD = max(Data(:,1)); nV = max(Data(:,2)); nL=max(Label(:,2));

val = ones(size(Data,1),1);
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
X=logical(X);
X=double(X);

val = ones(size(Label,1),1);
L=sparse(Label(:,1),Label(:,2),val,nD,nL);
L=logical(L);
L=double(L);

%% Results
%%% Tensor AUC=> 10:0.844003 20: 0.854408  50:0.846095 75:0.846213 100:0.846528 125: 0.848213 150:0.849047
%%% LEML AUC=>  10: 0.832948 20: 0.836863 50: 0.835706 75: 0.832998
%%% 100: 0.830738 125: 0.828738 150:0.827578
%%% 0.830738 
%%% Tensor Mem=> 4.326, 4.372, 4.385, 4.421, 4.471

%% Training on Words%%
t0=tic;
[nD nV]=size(X);
M1 = full(sum(X,1)');
Z1=sum(M1); M1=M1./Z1;

Xsum = full(sum(X,2));
Z2 = sum(Xsum.*Xsum);

M2=full(X'*X);
M2=M2./Z2;

K = 50; %Number of Clusters
[U,S]=eigs(M2,K);
s = diag(S);
W = U*diag(1./sqrt(s));
clearvars M2 U S;
s_M2=s;

Xsum = full(sum(X,2));
Z3 = sum(Xsum.*Xsum.*Xsum);
Mx=X*W./power(Z3,1/3);

G = zeros(K,K,K);
for i=1:K
    for j=1:K
        G(:,i,j)=(Mx(:,i).*Mx(:,j))'*Mx;
    end
    sprintf('Matrix Multiplied: %d',i)
end


% Extract tensor eigenvalues
G=tensor(G);
eigvals = zeros(K,1);
V=zeros(K,K);
for k=1:K
    G=symmetrize(G);
    [s,U]=sshopm(G,'Tol',1e-16);
    if(s>0) eigvals(k)=s; V(:,k)=U;
    else eigvals(k)=-s; V(:,k)=-U;
    end

    G=G-tensor(ktensor(s,U,U,U));
    fprintf(1,'%d th EigenValue Extracted: %f\n',k,s);
end

%clearvars G;

%W2 = W*inv(W'*W);
V2 = zeros(nV,K); W2 = pinv(W');
for k=1:K
    V2(:,k)=eigvals(k)*W2*V(:,k);
end

%% Moment on Labels %%
Ml = Mx*V;
Plabel = L'*(Ml.*Ml);
[i j v]=find(Plabel<0); Plabel(i,j)=0; %Set the negative values to zero

for k=1:K
    Plabel(:,k)=Plabel(:,k)/sum(Plabel(:,k));
end



%% Parameter Extraction for Test Set %%
Data=load('~/Documents/Data/MultiLabel/Delicious/Delicious_test_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;
val = ones(size(Data,1),1);
nD = max(Data(:,1)); 
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
X=logical(X); X=double(X);


P=normalize_cols(V2);
beta = .0001; P = P + beta*ones(size(P))/nV;
for k=1:K
    P(:,k)=P(:,k)./sum(P(:,k));
end

Pu = zeros(K,nD);
for u=1:nD
    [i j n]=find(X(u,:));
    L = P(j',:);
    Lprob = log(pi) + sum(log(L),1)';
    [~,imax]=max(Lprob);
    ll = Lprob-Lprob(imax);
    prob = exp(ll) + realmin('double');
    prob = prob/sum(prob);
    if ~isempty(find(isnan(prob),1))
        error('NaN in user probability');
    end
    Pu(:,u)=prob;
    if(mod(u,1000)==0) sprintf('User Probability Extracted: %d',u)
    end
end
  

toc(t0);

filename = sprintf('~/Documents/Data/MultiLabel/Delicious/WordProb_Bibtex_K%d.txt',K);
dlmwrite(filename,P','delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/Delicious/DocProb_Bibtex_K%d.txt',K);
dlmwrite(filename,Pu,'delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/Delicious/LabelProb_Bibtex_K%d.txt',K);
dlmwrite(filename,Plabel','delimiter',' ');
    
%% Test %%
Label=load('~/Documents/Data/MultiLabel/Delicious/Delicious_test_label.txt');
Label(:,1)=Label(:,1) + 1; Label(:,2)=Label(:,2)+1;

val = ones(size(Label,1),1);
Ltest=sparse(Label(:,1),Label(:,2),val,nD,nL);
Ltest=logical(Ltest); L=double(Ltest);

userCount=0;
M = [1,2,5,10,20,50];
sumAP=zeros(length(M),1); count=0; sumPrec = zeros(length(M),1); sumRecall=zeros(length(M),1);
for u=1:nD
    Pl_u = Plabel*Pu(:,u);
    [score,ID]=sort(Pl_u,'descend');
    score = score/sum(score);
    sel=find(Ltest(u,:));
    
    if ~isempty(sel)
        for l=1:length(M)

            AP=averagePrecisionAtK(sel,ID(1:M(l)),M(l));
            sumAP(l) = sumAP(l) + AP;

            prec = length( intersect(sel,ID(1:M(l))) )/M(l);
            sumPrec(l) = sumPrec(l)+prec;

            recall = length( intersect(sel,ID(1:M(l))) )/length(sel);
            sumRecall(l) = sumRecall(l) + recall;

            count = count+1;

            
        end
        
        
        userCount=userCount+1;
        if(l==length(M))% && mod(userCount,10)==0)
           fprintf(1,'%d: MAP:%f Precision:%f Recall:%f\n',userCount,sumAP(l)/userCount,sumPrec(l)/userCount,sumRecall(l)/userCount);
           
        end
        

        
    end
end
