## Polymorphism

Polymorphism means "many forms".  


### 1. Storage Service Application

A storage service can work with different storage providers  
through the same interface.  

The application does not care if files are stored:  

- locally
- in Amazon S3
- in Google GCS

It only knows: store(fileName)

At runtime, Java decides which implementation to execute.  

~~~java
package com.minte9.oop.polymorphism;

public class StorageServiceApp {
    public static void main(String[] args) {
        
        // Same interface reference
        StorageProvider storage = new LocalStorage();

        FileService service = new FileService(storage);
        service.uploadFile("photo.jpg");  // Saving file locally: photo.jpg

        // Runtime behavior changes
        storage = new CloudStorage();
        service = new FileService(storage);
        service.uploadFile("video.mp4");  // Uploading file to cloud: video.mp4
    }
}

interface StorageProvider {
    void storage(String fileName);    
}

// First implementation
class LocalStorage implements StorageProvider {
    @Override
    public void storage(String fileName) {
        System.out.println("Saving file locally: " + fileName);
    }
}

// Second implementation
class CloudStorage implements StorageProvider {
    @Override
    public void storage(String fileName) {
        System.out.println("Uploading file to cloud: " + fileName);
    }
}

// Composition
class FileService {

    // Dependency Inversion:
    private StorageProvider storage;  // Business logic depends on abstraction

    // Dependency Injection:
    public FileService(StorageProvider storage) {  // Dependency injectected from outside
        this.storage = storage;
    }

    // Polymorphism: 
    public void uploadFile(String fileName) {
        storage.storage(fileName);  // Same method call, different runtime behavior
    }
}
~~~


### 2. Dependency Injection

A beginner ofthen writes this:

~~~java
class FileService {

    private CloudStorage storage = new CloudStorage();
}
~~~

Problem:

- tightly coupled
- impossible to switch implementation
- difficult to test
- business logic controls low-level details

With Dependency Injection:

- implementation is external
- behavior is configurable
- dependencies are replaceable

~~~java
class FileService {

    private StorageProvider storage;  // Business logic depends on abstraction

    public FileService(StorageProvider storage) {  // Dependency injectected from outside
        this.storage = storage;
    }
}
~~~

Dependency Injection is everywhere in:

- Spring Framework
- Hibernate
- JUnit


### 3. Final Keyword

Suppose storaget provider should never change after service creation.  
That is a perfect use of final (cannot be reasigned, overriden, extended).    

~~~java
class FileService {
    private final StorageProvider storage;  // Look Here

    public FileService(StorageProvider storage) {
        this.storage = storage;
    }
}
~~~

Usefull when:

- security matters
- workflow must not change
- algorithm must stay consistent

Example:

- authentication flow
- transcation handling
- framework lifecycle methods

Extremely common, because dependencies should not mutate:

- Spring Framework
- Dependency Injection
- Clean arhitecture
- Immutable desing


### 4. Static Keyword

Suppose your application wants to count uploaded files globally.  
That is perfect use of static (belongs to the class/service).  

That count belongs to: 

- the whole application
- not to one FileService object

~~~java
class FileService {

    public static int totalUploads = 0;  // shared by all instances
    private StorageProvider storage;

    public FileService(StorageProvider storage) {
        this.storage = storage;
    }

    public void uploadFile(String fileName) {
        storage.store(fileName);
        totalUploads++;  // Look Here
    }
}
~~~