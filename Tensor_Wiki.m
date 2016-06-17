clear all;
Data=load('~/Documents/Data/MultiLabel/Wiki/wiki10_train_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;


nD = max(Data(:,1)); nV = max(Data(:,2)); 
fprintf(1,'No. of Documents %d, Vocabulary Size %d\n',nD,nV);

val = ones(size(Data,1),1);
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
clearvars Data;
X=logical(X);
X=double(X);

%% Results 
%%% Tensor AUC: 50: 0.910915, 75: 0.912114, 100: 0.912976, 125:0.913093, 150: 0.912786
%%% Memory: 50: 5257 75: 5365, 100: 5.411G (420 s), 125: 5.456 150: 5.496
%%% LEML Mem: 50: 2.27,75: 3.06,100: 4.129, 125: 4.343, 150: 4.375
%%% 2.91, 3.06, 3.24, 3.38, 3.498
%%% LEML AUC: 50:0.912863 , 75:0.910813 ,100: 0.906693, 125:0.903819, 150:0.902051     
%% Training on Words%%
t0=tic;
[nD nV]=size(X);
M1 = full(sum(X,1)');
Z1=sum(M1); M1=M1./Z1;

Xsum = full(sum(X,2));
Z2 = sum(Xsum.*Xsum);

%% Whitening
K = 150; %Number of Clusters


%[U,S]=eigs(full(X*X')/Z2,K);
[U,S,~]=svds(X/sqrt(Z2),K);
s=diag(S);
W=(X'*U*diag(1./(s.*s)))/sqrt(Z2);




%% Tensor Factorization
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

clearvars G;

%W2 = W*inv(W'*W);
V2 = zeros(nV,K); W2 = pinv(W');
for k=1:K
    V2(:,k)=eigvals(k)*W2*V(:,k);
end

clearvars X; 

%% Moment on Labels %%
Label=load('~/Documents/Data/MultiLabel/Wiki/wiki10_train_label.txt');
Label(:,1)=Label(:,1) + 1; Label(:,2)=Label(:,2)+1;
nL=max(Label(:,2));

fprintf(1,'No. of Documents %d, Label Size %d\n',nD,nL);

val = ones(size(Label,1),1);
L=sparse(Label(:,1),Label(:,2),val,nD,nL);
clearvars Label;
L=logical(L);
L=double(L);


Ml = Mx*V;
Plabel = L'*(Ml.*Ml);
[i j v]=find(Plabel<0); Plabel(i,j)=0; %Set the negative values to zero

for k=1:K
    Plabel(:,k)=Plabel(:,k)/sum(Plabel(:,k));
end

fprintf(1,'Training Finished\n');
toc(t0);

clearvars L; clearvars nD;

%% Parameter Extraction for Test Set %%
Data=load('~/Documents/Data/MultiLabel/Wiki/wiki10_test_word.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;
val = ones(size(Data,1),1);
nDtest = max(Data(:,1)); 
X=sparse(Data(:,1),Data(:,2),val,nDtest,nV);
clearvars Data;
X=logical(X); X=double(X);


P=normalize_cols(V2);
beta = .001; P = P + beta*ones(size(P))/nV;
for k=1:K
    P(:,k)=P(:,k)./sum(P(:,k));
end


Pu = zeros(K,nDtest);
for u=1:nDtest
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
    if(mod(u,1000)==0) fprintf(1,'Test Document Probability Extracted: %d\n',u); toc(t0);
    end
end
  

filename = sprintf('~/Documents/Data/MultiLabel/Wiki/WordProb_Wiki_K%d.txt',K);
dlmwrite(filename,P','delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/Wiki/DocProb_Wiki_K%d.txt',K);
dlmwrite(filename,Pu,'delimiter',' ');
filename = sprintf('~/Documents/Data/MultiLabel/Wiki/LabelProb_Wiki_K%d.txt',K);
dlmwrite(filename,Plabel','delimiter',' ');
    
