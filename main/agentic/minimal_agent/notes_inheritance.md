### Inheritance 

"Prefer composition/interfaces over inheritance" is a good advice most of the time.  
But inheritance is still the right tool when there's a true "is-a" relationship + shared state + shared base behavior.

### 1. Payment Processing System

We can have different payment methods:

- Credit card
- PayPal
- Bank transfer

They all:

- Have an amout
- Have a common processing flow
- But differ in how the payment is executed

~~~java
package com.minte9.oop.inheritance;

public class PaymentApp {
    public static void main(String[] args) {
        Payment p1 = new CreditCardPayment(100);
        Payment p2 = new PayPalPayment(200);

        p1.processPayment();
        p2.processPayment();

        /*
            Validating payment of $100.0
            Processing credit card payment...
            Sending receipt...

            Validating payment of $200.0
            Processing PayPal payment...
            Sending receipt...
        */
    }
}

// Superclass
abstract class Payment {
    protected double amount;

    public Payment(double amount) {
        this.amount = amount;
    }

    public void processPayment() {
        validate();
        execute();  // subclass-specific
        sendReceipt();
    }

    public void validate() {
        System.out.println("Validating payment of $" + amount);
    }

    protected abstract void execute();  // Look Here (must be implemented)

    public void sendReceipt() {
        System.out.println("Sending receipt...\n");
    }
}

// Subclass 1
class CreditCardPayment extends Payment {

    public CreditCardPayment(double amount) {
        super(amount);
    }

    @Override
    protected void execute() {
        System.out.println("Processing credit card payment...");
    }
}

// Subclass 2
class PayPalPayment extends Payment {

    public PayPalPayment(double amount) {
        super(amount);
    }

    @Override
    protected void execute() {
        System.out.println("Processing PayPal payment...");
    }
}
~~~

### 1.1 Why inheritance is the right choise here?

a) Strong "is-a" relationship:

- CreditCardPayment IS-a payment
- PayPalPayment IS-a payment

b) Shared logic that should NOT be duplicated.

You don't want every class reimplementing this.  

~~~java
processPayment()

// Validation
// Receipt sending
// Flow control
~~~

c) Sublcasses only customize:

~~~java
execute()
~~~

### 1.2 Why not use interfaces?

You could do:

~~~java
interface Payment() {
    void process();
}
~~~

But then:

- You lose shared state (amount)
- You duplicate workflow logic
- You can't enforce the sequence (validate -> execute -> receipt)

### 1.3 Rule of thumb about inheritance

Use inheritance when:

- There is a true hierachy (is-a)
- You need shared state
- You want to enforce a common algorithm/flow
- Subclasses only tweak specific steps

Use interfaces when:

- You want capabilities/roles (Serializable, Runnable)
- Behavior varies wildly
- No shared implementation is needed

### 1.4 Payment system (done wrong)

A developer might try to be "modern" and do everything with interfaces. 

~~~java
package com.minte9.oop.inheritance.wrong_implementation;

public class PaymentWrongApp {
    public static void main(String[] args) {
        Payment p1 = new CreditCardPayment(100);
        Payment p2 = new PayPalPayment(200);

        p1.processPayment();
        p2.processPayment();

        /*
            Validating $100.0
            Processing credit card...
            Sending receipt...

            Validating $200.0
            Processing PayPal...
            Sending receipt...
        */
    }
}

interface Payment  {
    void processPayment();    
}

class CreditCardPayment implements Payment {
    private double amount;

    public CreditCardPayment(double amount) {
        this.amount = amount;
    }

    @Override
    public void processPayment() {
        System.out.println("Validating $" + amount);
        System.out.println("Processing credit card...");
        System.out.println("Sending receipt...\n");
    }
}

class PayPalPayment implements Payment {
    private double amount;

    public PayPalPayment(double amount) {
        this.amount = amount;
    }

    @Override
    public void processPayment() {
        System.out.println("Validating $" + amount);
        System.out.println("Processing PayPal...");
        System.out.println("Sending receipt...\n");
    }
}
~~~

### 1.5 What's wrong here?

a) Massive duplication, every class repeats:

~~~java
System.out.println("Validating $" + amount);
System.out.println("Sending receipt...\n");
~~~

If you change validation logic, you must update every class.

b) No shared state

Every class defines amount, but that's cleary common.

c) No control over workflow

Nothing enforces:

~~~sh
validate -> execute -> sendRecipt
~~~

A developer could accidentally do:

~~~java
sendReceipt();
execute();
~~~

### 2. Abstract Keyword

An abstract class CANNOT be instantiated directly.

What exactly is a Payment?

- Is it a credit card payment? A PayPal payment? A bank transfer?
- We don't know, so Payment should be abstract.

An abstract class is meant to be extended by subclasses.  
An abstract method defines behavior that subclasses must implement.  

~~~java
package com.minte9.oop.inheritance.abstract_note;

public class AbstractKeyword {

    public static void main(String[] args) {

        CreditCardPayment payment = new CreditCardPayment();
        
        payment.setAmount(250);
        payment.processPayment();  // Processing credit card payment of $250.0
    }
}

abstract class Payment {
    protected double amount;

    public abstract void processPayment();  // abstract (must be implemented)

    public void setAmount(double amount) {  // non-abstract method (shared behavior)
        this.amount = amount;
    }
}

class CreditCardPayment extends Payment {

    @Override
    public void processPayment() {
        System.out.println("Processing credit card payment of $" + amount);
    }
}
~~~

### 3. Override Annotation

The @Override annotation acts as a compile-time safeguard.  

It tells the compiler:  
This method must override a method from the parent class.  

Without @Override, a typo creates a new method instead of overriding,  
and the error may go unnoticed until runtime behavior is wrong.  

~~~java
package com.minte9.oop.inheritance.override_note;

public class OverrideAnnotation {

    public static void main(String[] args) {
        
        CreditCardPayment payment = new CreditCardPayment();
        payment.processPayment();  // Wrong method called!
    }
}

abstract class Payment {
    public abstract void processPayment();
}

class CreditCardPayment extends Payment {

    // Typo: processsPayment instead of processPayment
    // No @Override annotation to protect us
    public void processsPayment() {
        System.out.println("Processing credit card payment...");
    }

    // This method was auto-generated later just to satisfy compilation
    @Override
    public void processPayment() {
        System.out.println("Wrong method called!");
    }
}
~~~