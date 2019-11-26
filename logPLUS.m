function logZ = logPLUS(logX,logY)

if isinf(logX) && isinf(logY)
    logZ = -inf;
    return;
end

if logX > logY
    logZ = logX+log1p(exp(logY-logX));
else
    logZ = logY+log1p(exp(logX-logY));
end


end
