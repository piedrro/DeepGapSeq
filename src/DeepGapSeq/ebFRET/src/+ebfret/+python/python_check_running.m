function [running] = check_running(self)
    
    running = exist('ebfret', 'file') == 2;

end