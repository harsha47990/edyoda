 --question 1
select count(*) from SalesPeople where Sname like '%a'
 
 --question 2
select s.Sname from SalesPeople as s
inner join Orders as o on s.Snum = o.Snum where o.Amt > 2000
 
 --question 3
select count(*) from SalesPeople where City = 'Newyork'
 
 --question 4
select Sname from SalesPeople where city in ('London','Paris')
 
 --question 5
select Count(o.Onum), s.Sname, o.Odate from Orders as o 
inner join SalesPeople as s group by s.Sname, o.Odate
