%Pattern Classification:
size = 10000;
PTrain = csvread('train.csv',1,1);
PTest = csvread('test.csv',1,1);

PTrain = PTrain(1:size,:);
TTrain = PTrain(:,1)  ;
PTrain(:,1) = [] ;%Remove the targets

ratio = 2;
Psize = round(size/ratio);
Label = threeParts(PTrain,TTrain,PTest,Psize);
%Label(Label == 10) = 0;  % change '10' to '0'
n = 10000;
ImageId = 1:n; ImageId = ImageId';                  % image ids
writetable(table(ImageId, Label), 'submission.csv');% write to csv