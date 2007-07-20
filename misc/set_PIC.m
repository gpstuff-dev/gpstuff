function  [blocks, Xu] = set_PIC(x, dims, cellsize, blocksize, indtype, visualize)
 

% define the blocks by cubes! Total 9 cubes

ly = dims(2)/blocksize;
lx = dims(4)/blocksize;

b1 = linspace(0,dims(4),lx);
b2 = linspace(0,dims(2),ly);
index={}; indexsize = [];
for i1=1:length(b1)-1
    for i2=1:length(b2)-1
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b2(i2)<=x(ind',2) & x(ind',2) < b2(i2+1));
        if ~isempty(ind)
            index{end+1} = ind';
            indexsize(end+1) = length(ind);
        end
        %            plot(x(ind,1),x(ind,2),col{i1,i2})        
    end
end
minblock = min(indexsize); maxblock = max(indexsize); avgblock = mean(indexsize);

switch indtype
  case 'corners'
    [xii,yii]=meshgrid(b1,b2);
    xii=[xii(:) yii(:)];
    qm=min(sqrt(gminus(x(:,1),xii(:,1)').^2+gminus(x(:,2),xii(:,2)').^2));
    qii=qm<=blocksize/2;sum(qii);
    Xu = unique(xii(qii,:), 'rows');
  case 'corners+1xside'
    b12 = linspace((b1(2)-b1(1))/2,dims(4)+(b1(2)-b1(1))/2,lx);
    b22 = linspace((b2(2)-b2(1))/2,dims(2)+(b2(2)-b2(1))/2,ly);

    [xii,yii]=meshgrid(b1,b2);
    xii=[xii(:) yii(:)];
    [xii2,yii2]=meshgrid(b12,b2);
    xii2=[xii2(:) yii2(:)];
    xii = [xii;xii2];
    [xii2,yii2]=meshgrid(b1,b22);
    xii2=[xii2(:) yii2(:)];
    xii = [xii;xii2];
    qm=min(sqrt(gminus(x(:,1),xii(:,1)').^2+gminus(x(:,2),xii(:,2)').^2));
    qii=qm<=blocksize/2;sum(qii);
    Xu = unique(xii(qii,:), 'rows');
end

numu = length(Xu);
% $$$ D = 0;
% $$$ for i1 = 1:size(Xu,2);
% $$$     D = D + gminus(Xu(:,i1), Xu(:,i1)').^2;
% $$$ end
% $$$ meandist = mean(mean(sqrt(D)));

% Include the blocks with too few data points into larger ones
go = 1; i1=1;
while go == 1
    if length(index{i1}) < maxblock/2
        meandist=[];
        if i1>1
            others = 1:i1-1;
        else
            others = [];
        end
        if i1 < length(index)
            others = [others i1+1:length(index)];
        end;
        for i2 = 1:length(others)
            D = 0;
            for i3 = 1:size(x,2);
                D = D + gminus(x(index{i1},i3), x(index{others(i2)},i3)').^2;
            end
            meandist(i2) = mean(mean(sqrt(D)));
        end
        minmean = others(find(meandist == min(meandist)));
        index{i1} =  sort([index{i1};index{minmean}]);
        if minmean==1
            index = {index{2:end}};
        elseif minmean==length(index)
            index = {index{1:end-1}};
        else
            index = {index{1:minmean-1} index{minmean+1:end}};
        end
    else
        i1 = i1+1;
    end
    if i1 > length(index)
        go = 0;
    end
end

for i1 = 1:length(index)
    indexsize2(i1) = length(index{i1});
end
minblock = min(indexsize2); maxblock = max(indexsize2); avgblock = mean(indexsize2);


if visualize == 1
        
     col = {'b.','g.','c.','m.','y.','k.','b.','g.','c.','m.';
          'g.','c.','m.','y.','k.','b.','g.','c.','m.','b.';
          'c.','m.','y.','k.','b.','g.','c.','m.','b.','g.';
          'm.','y.','k.','b.','g.','c.','m.','b.','g.','c.';
          'y.','k.','b.','g.','c.','m.','b.','g.','c.','m.';
          'k.','b.','g.','c.','m.','b.','g.','c.','m.','y.';
          'b.','g.','c.','m.','b.','g.','c.','m.','y.','k.';};

    
    figure 
    set(gcf,'units','centimeters');
    hold on
    for i1=1:length(index)
        plot(x(index{i1},1),x(index{i1},2),'.', 'Color', [i1./length(index) abs(sin(i1)) 1/i1])
    end
    plot(Xu(:,1),Xu(:,2), 'r*', 'MarkerSize', 8, 'LineWidth', 2)

    tmpy = 1:dims(2);
    tmpx = 1:dims(4);
    for i1 = 1:length(b1)
        plot(repmat(b1(i1), 1, length(tmpy)), tmpy,'k--')
    end
    for i1 = 1:length(b2)
        plot(tmpx, repmat(b2(i1), 1, length(tmpx)),'k--')
    end
    axis equal
    xlim([0 dims(4)])
    ylim([0 dims(2)])
    set(gcf, 'pos', [11.3896   15.9581   15   25])
    
    S = sprintf('%d blocks, with %.f data points in averige and %d/%d at most/least \n %d inducing inputs.', ...
                length(index), avgblock, maxblock, minblock, numu);
    H = text(-5, -10, S);
    set(H, 'FontSize', 14);
end

blocks = index;

% $$$ switch cellsize
% $$$   case 20000
% $$$     index2{1} = index{1};
% $$$     index2{2} = index{2};
% $$$     index2{3} = sort([index{3};index{7};index{11}]);
% $$$     index2 = {index2{:} index{4:6} index{8:10}};
% $$$ 
% $$$     col = {'b*','g*','c*','m*','y*','k*','b*','g*','m*'};
% $$$     figure, hold on
% $$$     for i1=1:9
% $$$         plot(x(index2{i1},1),x(index2{i1},2),col{i1})
% $$$     end
% $$$     plot(U(:,1),U(:,2),'r*')
% $$$ 
% $$$ 
% $$$ % Set the inducing inputs 
% $$$ num = 4; num2=4;
% $$$ [xii,yii]=meshgrid(0:num:36,0:num2:60);
% $$$ xii=[xii(:) yii(:)];
% $$$ qm=min(sqrt(gminus(x(:,1),xii(:,1)').^2+gminus(x(:,2),xii(:,2)').^2));
% $$$ qii=qm<=1;sum(qii);
% $$$ U = xii(qii,:);
% $$$ 
% $$$ [n,nin] = size(x);
