function genSubmission(class, str)

headers = {'ImageId','Label'};
submission = NaN(28000,2);
submission(1:28000,2) = class;
submission(1:28000,1) = [1:28000];
csvwrite_with_headers(sprintf('submission/submission_%s.csv', str),submission,headers);

